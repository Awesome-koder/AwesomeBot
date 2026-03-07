import html
import logging
import re
import time
from typing import Optional
from datetime import datetime
import streamlit as st
from langchain_core.documents import Document

from app_core import (
    answer_question,
    build_vector_store,
    extract_text_from_pdfs,
    fingerprint_uploads,
    resolve_api_key,
    split_text,
)

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

CHAT_HISTORY_KEY = "chat_history"
MESSAGE_COUNTER_KEY = "message_counter"
PENDING_RETRY_KEY = "pending_retry_id"
THEME_KEY = "theme_mode"
DEFAULT_ASSISTANT_MESSAGE = (
    "Upload one or more PDFs from the sidebar, click Process PDFs, then ask a question "
    "or generate a summary here."
)


def show_user_error(message: str):
    st.error(message)


def _ensure_api_key() -> bool:
    try:
        resolve_api_key(None)
        return True
    except ValueError:
        logger.exception("API key resolution failed")
        show_user_error("API key is missing. Add your configuration and try again.")
        return False


def _is_not_in_context(answer_text: str) -> bool:
    return "answer is not available in the context" in (answer_text or "").lower()


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _normalize_answer(text: str) -> str:
    return " ".join((text or "").lower().split())


def _split_into_segments(text: str):
    raw_parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", text or "")
    segments = []
    for part in raw_parts:
        cleaned = part.strip(" -\t")
        if cleaned:
            segments.append(cleaned)
    return segments


def _extract_question_focus(question: str) -> str:
    q = (question or "").strip().lower()
    patterns = [
        r"^what is\s+(.+?)[\?\.]?$",
        r"^who is\s+(.+?)[\?\.]?$",
        r"^define\s+(.+?)[\?\.]?$",
        r"^explain\s+(.+?)[\?\.]?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, q)
        if match:
            focus = match.group(1).strip(" .?")
            if focus:
                return focus
    return ""


def _polish_summary_answer(answer: str) -> str:
    text = " ".join((answer or "").split())
    if not text:
        return ""
    text = re.sub(r"^[A-Z][A-Z\s]{3,40}\s+(?=[A-Z][a-z])", "", text).strip()
    first_sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    if len(first_sentence) >= 30:
        text = first_sentence
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _is_low_quality_answer(answer: str, question: str = "") -> bool:
    text = " ".join((answer or "").split()).strip()
    if not text:
        return True

    if len(text) < 35:
        return True

    lowered = text.lower()
    if re.search(r"\b(is|are|was|were)\s+(a|an|the)?\s*$", lowered):
        return True

    if re.search(r"\b(overview|key points)\b", lowered):
        return True

    if re.search(r"(https?://|www\.|@|\+\d{1,3}\s?\d+)", lowered):
        return True

    focus = _extract_question_focus(question)
    if focus:
        focus_tokens = set(_tokenize(focus))
        answer_tokens = set(_tokenize(text))
        if focus_tokens and len(focus_tokens & answer_tokens) == 0:
            return True

    return False


def _answer_from_summary_text(question: str, summary_text: str) -> str:
    stopwords = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "did",
        "about",
    }

    query_terms = {t for t in _tokenize(question) if t not in stopwords}
    if not query_terms:
        return ""

    segments = _split_into_segments(summary_text)
    if not segments:
        return ""

    focus = _extract_question_focus(question)
    focus_tokens = set(_tokenize(focus))
    candidates = []

    for line in segments:
        line_tokens = set(_tokenize(line))
        if not line_tokens:
            continue

        overlap = len(query_terms & line_tokens)
        if overlap == 0:
            continue

        score = overlap / max(len(query_terms), 1)
        lowered = line.lower()
        if focus and focus in lowered:
            score += 0.8
        if focus_tokens and len(focus_tokens & line_tokens) == len(focus_tokens):
            score += 0.5
        if focus and (lowered.startswith(focus) or f"{focus} is " in lowered):
            score += 0.6
        if " is " in f" {lowered} ":
            score += 0.15
        if re.search(r"(https?://|www\.|@|\+\d{1,3}\s?\d+)", lowered):
            score -= 0.8
        if len(line) > 220:
            score -= 0.05

        candidates.append((score, line))

    if not candidates:
        return ""

    for score, line in sorted(candidates, key=lambda x: x[0], reverse=True):
        if score < 0.35:
            break
        polished = _polish_summary_answer(line)
        if not _is_low_quality_answer(polished, question):
            return polished

    return ""


def _build_retry_prompts(question: str, retry_index: int):
    if retry_index <= 0:
        return [question]

    return [
        (
            f"{question}\n\n"
            "Answer again using different wording and structure than the previous reply. "
            "Stay faithful to the uploaded document."
        ),
        (
            f"{question}\n\n"
            "Provide another concise answer from a different angle. "
            "Do not repeat the previous phrasing."
        ),
        (
            f"{question}\n\n"
            "Give an alternative explanation based only on the uploaded PDFs. "
            "Use a fresh phrasing and sentence structure."
        ),
    ]


def _make_answer_distinct(answer: str) -> str:
    clean_answer = (answer or "").strip()
    if not clean_answer:
        return clean_answer
    if clean_answer.lower().startswith("alternative phrasing:"):
        return clean_answer + " Please verify against the PDF text."
    return f"Alternative phrasing: {clean_answer}"


class SummaryVectorStore:
    def __init__(self, summary_text: str):
        self.summary_text = summary_text

    def similarity_search(self, _query: str, k: int = 1):
        if not self.summary_text.strip():
            return []
        return [Document(page_content=self.summary_text)]


def _init_session_state():
    defaults = {
        CHAT_HISTORY_KEY: [
            {
                "id": 1,
                "role": "assistant",
                "content": DEFAULT_ASSISTANT_MESSAGE,
                "source": "system",
                "allow_retry": False,
                "prompt": None,
                "retry_count": 0,
            }
        ],
        MESSAGE_COUNTER_KEY: 1,
        PENDING_RETRY_KEY: None,
        "vector_store": None,
        "doc_fingerprint": None,
        "doc_chunks": 0,
        "latest_summary": "",
        "last_uploaded_names": [],
        THEME_KEY: "light",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _next_message_id() -> int:
    st.session_state[MESSAGE_COUNTER_KEY] += 1
    return st.session_state[MESSAGE_COUNTER_KEY]


def _append_chat_message(
    role: str,
    content: str,
    source: Optional[str] = None,
    allow_retry: bool = False,
    prompt: Optional[str] = None,
    retry_count: int = 0,
):
    message = {
        "id": _next_message_id(),
        "role": role,
        "content": content,
        "source": source,
        "allow_retry": allow_retry,
        "prompt": prompt,
        "retry_count": retry_count,
    }
    st.session_state[CHAT_HISTORY_KEY].append(message)
    return message


def _render_source_caption(source: Optional[str] = None):
    if source == "summary":
        st.caption("Source: document summary")
    elif source == "index":
        st.caption("Source: indexed chunks")
    elif source == "local-summary":
        st.caption("Source: local extractive summary")
    elif source == "system":
        st.caption("Source: assistant guidance")


def _find_message_by_id(message_id: int):
    for message in st.session_state.get(CHAT_HISTORY_KEY, []):
        if message.get("id") == message_id:
            return message
    return None


def _render_chat_history():
    for message in st.session_state.get(CHAT_HISTORY_KEY, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_source_caption(message.get("source"))
            if (
                message["role"] == "assistant"
                and message.get("allow_retry")
                and message.get("prompt")
            ):
                st.button(
                    "Retry",
                    key=f"retry_{message['id']}",
                    on_click=_queue_retry,
                    args=(message["id"],),
                )


def _stream_assistant_message(
    content: str,
    source: Optional[str] = None,
    allow_retry: bool = False,
    prompt: Optional[str] = None,
    retry_count: int = 0,
):
    with st.chat_message("assistant"):
        placeholder = st.empty()
        rendered = ""
        tokens = re.findall(r"\S+\s*", content or "")
        if not tokens:
            placeholder.markdown("")
        else:
            for token in tokens:
                rendered += token
                placeholder.markdown(rendered + "▌")
                time.sleep(0.015)
            placeholder.markdown(rendered)
        _render_source_caption(source)
    _append_chat_message(
        "assistant",
        content,
        source=source,
        allow_retry=allow_retry,
        prompt=prompt,
        retry_count=retry_count,
    )


def _scroll_chat_to_bottom():
    st.markdown(
        """
        <div id="chat-bottom-anchor"></div>
        <script>
        const anchor = window.parent.document.getElementById("chat-bottom-anchor");
        if (anchor) {
            anchor.scrollIntoView({behavior: "smooth", block: "end"});
        }
        </script>
        """,
        unsafe_allow_html=True,
    )


def _queue_retry(message_id: int):
    st.session_state[PENDING_RETRY_KEY] = message_id


def _toggle_theme():
    current_theme = st.session_state.get(THEME_KEY, "light")
    st.session_state[THEME_KEY] = "dark" if current_theme == "light" else "light"


def _theme_tokens() -> dict[str, str]:
    theme = st.session_state.get(THEME_KEY, "light")
    if theme == "dark":
        return {
            "app_bg": "radial-gradient(circle at top, #172554 0%, #0f172a 42%, #020617 100%)",
            "panel_bg": "rgba(8, 15, 32, 0.9)",
            "panel_border": "rgba(148, 163, 184, 0.22)",
            "text_main": "#f8fafc",
            "text_soft": "#cbd5e1",
            "accent": "#38bdf8",
            "accent_soft": "rgba(56, 189, 248, 0.18)",
            "toolbar_bg": "rgba(8, 15, 32, 0.74)",
            "chat_input_bg": "rgba(8, 15, 32, 0.96)",
            "footer_bg": "rgba(2, 6, 23, 0.98)",
            "sidebar_bg": "linear-gradient(180deg, #0f172a 0%, #111827 100%)",
            "sidebar_border": "rgba(148, 163, 184, 0.18)",
            "message_user": "rgba(8, 47, 73, 0.72)",
            "message_assistant": "rgba(15, 23, 42, 0.92)",
            "composer_border": "rgba(56, 189, 248, 0.26)",
            "footer_link": "#7dd3fc",
            "status_bg": "rgba(15, 23, 42, 0.82)",
            "button_bg": "#e2e8f0",
            "button_text": "#0f172a",
            "button_border": "rgba(226, 232, 240, 0.7)",
            "button_hover": "#f8fafc",
            "button_secondary_bg": "rgba(15, 23, 42, 0.88)",
            "button_secondary_text": "#e2e8f0",
            "input_surface": "rgba(15, 23, 42, 0.92)",
            "input_border": "rgba(148, 163, 184, 0.28)",
            "placeholder": "#94a3b8",
            "dropzone_bg": "rgba(15, 23, 42, 0.86)",
            "dropzone_border": "rgba(56, 189, 248, 0.24)",
            "shadow": "0 16px 45px rgba(2, 6, 23, 0.42)",
        }
    return {
        "app_bg": "linear-gradient(180deg, #f8fafc 0%, #e0f2fe 48%, #eef2ff 100%)",
        "panel_bg": "rgba(255, 255, 255, 0.88)",
        "panel_border": "rgba(15, 23, 42, 0.08)",
        "text_main": "#0f172a",
        "text_soft": "#475569",
        "accent": "#0f766e",
        "accent_soft": "rgba(15, 118, 110, 0.12)",
        "toolbar_bg": "rgba(255, 255, 255, 0.74)",
        "chat_input_bg": "rgba(255, 255, 255, 0.97)",
        "footer_bg": "rgba(15, 23, 42, 0.96)",
        "sidebar_bg": "linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%)",
        "sidebar_border": "rgba(15, 23, 42, 0.08)",
        "message_user": "rgba(15, 118, 110, 0.08)",
        "message_assistant": "rgba(255, 255, 255, 0.9)",
        "composer_border": "rgba(15, 23, 42, 0.08)",
        "footer_link": "#60a5fa",
        "status_bg": "rgba(255, 255, 255, 0.78)",
        "button_bg": "#0f172a",
        "button_text": "#f8fafc",
        "button_border": "rgba(15, 23, 42, 0.12)",
        "button_hover": "#1e293b",
        "button_secondary_bg": "rgba(255, 255, 255, 0.92)",
        "button_secondary_text": "#0f172a",
        "input_surface": "rgba(255, 255, 255, 0.95)",
        "input_border": "rgba(15, 23, 42, 0.12)",
        "placeholder": "#64748b",
        "dropzone_bg": "rgba(255, 255, 255, 0.86)",
        "dropzone_border": "rgba(15, 23, 42, 0.14)",
        "shadow": "0 18px 48px rgba(15, 23, 42, 0.14)",
    }


def _reset_document_state():
    st.session_state["vector_store"] = None
    st.session_state["doc_fingerprint"] = None
    st.session_state["doc_chunks"] = 0
    st.session_state["latest_summary"] = ""


def render_sidebar():
    st.sidebar.title("PDF Processing")

    pdf_docs = st.sidebar.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    max_chunks = st.sidebar.number_input(
        "Max chunks",
        min_value=10,
        max_value=500,
        value=120,
        step=10,
        help="Limit chunks to control cost and latency.",
    )

    process_clicked = st.sidebar.button("Process PDFs", type="secondary")

    return pdf_docs, max_chunks, process_clicked


def build_and_cache_vector_store(pdf_docs, max_chunks):
    if not pdf_docs:
        show_user_error("Please upload at least one PDF file to continue.")
        return False

    if not _ensure_api_key():
        return False

    try:
        fingerprint = fingerprint_uploads(pdf_docs)
    except Exception:
        logger.exception("Failed to fingerprint uploads")
        show_user_error("We could not identify the uploaded files. Please try again.")
        return False

    if (
        st.session_state.get("vector_store") is not None
        and st.session_state.get("doc_fingerprint") == fingerprint
    ):
        st.sidebar.success("Using cached vector store for the current PDFs.")
        return True

    try:
        with st.spinner("Extracting text..."):
            raw_text, file_errors = extract_text_from_pdfs(pdf_docs)
    except Exception:
        logger.exception("PDF text extraction failed")
        show_user_error("We could not read the uploaded files right now. Please try again.")
        return False

    if file_errors:
        st.warning("Some files could not be processed and were skipped.")
        for error in file_errors:
            logger.warning(error)

    if not raw_text:
        show_user_error("No readable text was found in the uploaded PDF files.")
        return False

    try:
        with st.spinner("Chunking text..."):
            chunks = split_text(raw_text)
    except Exception:
        logger.exception("Text chunking failed")
        show_user_error("We could not prepare your document for search. Please try again.")
        return False

    if not chunks:
        show_user_error("Document processing produced no usable content. Try a different PDF.")
        return False

    chunks = chunks[:max_chunks]

    with st.spinner("Building vector store..."):
        try:
            vector_store = build_vector_store(chunks)
        except Exception:
            logger.exception("Vector store build failed")
            show_user_error("We could not index your document right now. Please retry shortly.")
            return False

    st.session_state["vector_store"] = vector_store
    st.session_state["doc_fingerprint"] = fingerprint
    st.session_state["doc_chunks"] = len(chunks)
    st.session_state["last_uploaded_names"] = [doc.name for doc in pdf_docs]
    st.session_state["latest_summary"] = ""
    st.success("PDFs processed. You can ask questions now.")
    return True


def handle_question(
    question,
    previous_answers=None,
    retry_index: int = 0,
):
    clean_question = (question or "").strip()
    if not clean_question:
        return {"ok": False, "message": "Please enter a question.", "kind": "warning"}

    if len(clean_question) < 4:
        return {"ok": False, "message": "Question is too short.", "kind": "warning"}

    if st.session_state.get("vector_store") is None:
        return {
            "ok": False,
            "message": "Process PDFs before asking questions.",
            "kind": "info",
        }

    previous_answers = previous_answers or []
    previous_norms = {_normalize_answer(item) for item in previous_answers if item}

    answer = None
    source = "index"
    summary_text = (st.session_state.get("latest_summary") or "").strip()

    if summary_text and retry_index == 0:
        direct_summary_answer = _answer_from_summary_text(clean_question, summary_text)
        if direct_summary_answer and not _is_low_quality_answer(
            direct_summary_answer, clean_question
        ):
            answer = direct_summary_answer
            source = "summary"
            logger.info("Answered directly from summary text.")

    if summary_text and answer is None:
        if not _ensure_api_key():
            return {
                "ok": False,
                "message": "API key is missing. Add your configuration and try again.",
                "kind": "error",
            }
        try:
            summary_store = SummaryVectorStore(summary_text)
            for prompt_variant in _build_retry_prompts(clean_question, retry_index):
                with st.spinner("Searching summary..."):
                    summary_answer = answer_question(summary_store, prompt_variant, k=1)
                polished_summary_answer = _polish_summary_answer(summary_answer)
                if (
                    not _is_not_in_context(polished_summary_answer)
                    and not _is_low_quality_answer(polished_summary_answer, clean_question)
                    and _normalize_answer(polished_summary_answer) not in previous_norms
                ):
                    answer = polished_summary_answer
                    source = "summary"
                    break
        except Exception:
            logger.exception("Summary-first lookup failed, falling back to index chunks")

    if answer is None:
        if not _ensure_api_key():
            return {
                "ok": False,
                "message": "API key is missing. Add your configuration and try again.",
                "kind": "error",
            }
        try:
            for prompt_variant in _build_retry_prompts(clean_question, retry_index):
                with st.spinner("Searching indexed content..."):
                    candidate = answer_question(st.session_state["vector_store"], prompt_variant)
                if _normalize_answer(candidate) not in previous_norms:
                    answer = candidate
                    source = "index"
                    break
            if answer is None:
                with st.spinner("Searching indexed content..."):
                    answer = answer_question(st.session_state["vector_store"], clean_question)
                source = "index"
        except Exception:
            logger.exception("Question answering failed")
            return {
                "ok": False,
                "message": (
                    "We could not generate an answer right now. Please try again in a moment."
                ),
                "kind": "error",
            }

    if previous_norms and _normalize_answer(answer) in previous_norms:
        answer = _make_answer_distinct(answer)

    return {"ok": True, "message": answer, "source": source}


def handle_summary():
    if st.session_state.get("vector_store") is None:
        return {
            "ok": False,
            "message": "Process PDFs before generating a summary.",
            "kind": "info",
        }

    if not _ensure_api_key():
        return {
            "ok": False,
            "message": "API key is missing. Add your configuration and try again.",
            "kind": "error",
        }

    summary_prompt = (
        "Generate a clear summary of the uploaded PDF content. "
        "Use concise sections with: Overview, Key Points, and Important Details."
    )
    source = "summary"
    try:
        with st.spinner("Generating summary..."):
            summary = answer_question(st.session_state["vector_store"], summary_prompt)
    except Exception:
        logger.exception("Summary generation failed; attempting local fallback summary")
        summary = build_local_summary(st.session_state["vector_store"])
        if not summary:
            return {
                "ok": False,
                "message": (
                    "We could not generate a summary right now. Please try again in a moment."
                ),
                "kind": "error",
            }
        source = "local-summary"

    st.session_state["latest_summary"] = summary
    return {"ok": True, "message": summary, "source": source}


def build_local_summary(vector_store, k: int = 8):
    try:
        docs = vector_store.similarity_search(
            "overall document summary key points important details", k=k
        )
    except Exception:
        logger.exception("Local summary retrieval failed")
        return ""

    snippets = []
    for doc in docs:
        text = getattr(doc, "page_content", "") or ""
        cleaned = " ".join(text.split())
        if cleaned:
            snippets.append(cleaned[:280])

    if not snippets:
        return ""

    unique = []
    seen = set()
    for snippet in snippets:
        if snippet in seen:
            continue
        seen.add(snippet)
        unique.append(snippet)
        if len(unique) >= 5:
            break

    lines = ["Overview:", "This is a quick extractive summary from the uploaded document."]
    lines.append("")
    lines.append("Key points:")
    for item in unique:
        lines.append(f"- {item}")
    return "\n".join(lines)


def _show_status_message(result):
    level = result.get("kind", "error")
    message = result["message"]
    if level == "info":
        st.info(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)


def _handle_user_prompt(prompt: str):
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        st.warning("Please enter a question.")
        return

    _append_chat_message("user", clean_prompt)
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    result = handle_question(clean_prompt)
    if result["ok"]:
        _stream_assistant_message(
            result["message"],
            source=result.get("source"),
            allow_retry=True,
            prompt=clean_prompt,
            retry_count=0,
        )
    else:
        _show_status_message(result)


def _handle_retry_request():
    message_id = st.session_state.get(PENDING_RETRY_KEY)
    if not message_id:
        return

    st.session_state[PENDING_RETRY_KEY] = None
    original_message = _find_message_by_id(message_id)
    if not original_message:
        return

    prompt = original_message.get("prompt")
    if not prompt:
        return

    previous_answers = [
        item["content"]
        for item in st.session_state.get(CHAT_HISTORY_KEY, [])
        if item.get("role") == "assistant" and item.get("prompt") == prompt
    ]

    retry_count = int(original_message.get("retry_count", 0)) + 1
    result = handle_question(
        prompt,
        previous_answers=previous_answers,
        retry_index=retry_count,
    )
    if result["ok"]:
        _stream_assistant_message(
            result["message"],
            source=result.get("source"),
            allow_retry=True,
            prompt=prompt,
            retry_count=retry_count,
        )
    else:
        _show_status_message(result)


def _render_header():
    st.title("PDF Reader AI")
    st.caption(
        "Upload PDFs, build a searchable index, generate a summary, and chat with your documents."
    )


def _build_app_styles() -> str:
    theme = st.session_state.get(THEME_KEY, "light")
    tokens = _theme_tokens()
    return f"""
        <style>
        :root {{
            --app-bg: {tokens['app_bg']};
            --panel-bg: {tokens['panel_bg']};
            --panel-border: {tokens['panel_border']};
            --text-main: {tokens['text_main']};
            --text-soft: {tokens['text_soft']};
            --accent: {tokens['accent']};
            --accent-soft: {tokens['accent_soft']};
            --toolbar-bg: {tokens['toolbar_bg']};
            --chat-input-bg: {tokens['chat_input_bg']};
            --footer-bg: {tokens['footer_bg']};
            --sidebar-bg: {tokens['sidebar_bg']};
            --sidebar-border: {tokens['sidebar_border']};
            --message-user-bg: {tokens['message_user']};
            --message-assistant-bg: {tokens['message_assistant']};
            --composer-border: {tokens['composer_border']};
            --footer-link: {tokens['footer_link']};
            --status-bg: {tokens['status_bg']};
            --button-bg: {tokens['button_bg']};
            --button-text: {tokens['button_text']};
            --button-border: {tokens['button_border']};
            --button-hover: {tokens['button_hover']};
            --button-secondary-bg: {tokens['button_secondary_bg']};
            --button-secondary-text: {tokens['button_secondary_text']};
            --input-surface: {tokens['input_surface']};
            --input-border: {tokens['input_border']};
            --placeholder: {tokens['placeholder']};
            --dropzone-bg: {tokens['dropzone_bg']};
            --dropzone-border: {tokens['dropzone_border']};
            --shadow: {tokens['shadow']};
            --content-max-width: 980px;
            --sidebar-width: 19rem;
            --bottom-footer-height: 4.9rem;
        }}

        .stApp, [data-testid="stAppViewContainer"], .main {{
            background: var(--app-bg);
            color: var(--text-main);
        }}

        [data-testid="stAppViewContainer"] {{
            padding-bottom: 9rem;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        .block-container {{
            max-width: var(--content-max-width);
            padding-top: 1.25rem;
            padding-bottom: 10rem;
        }}

        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg);
            border-right: 1px solid var(--sidebar-border);
            height: 100dvh;
        }}

        [data-testid="stSidebar"] > div {{
            height: 100%;
        }}

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] *,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] *,
        [data-testid="stText"],
        label,
        p,
        span,
        h1, h2, h3 {{
            color: var(--text-main) !important;
        }}

        [data-testid="stCaptionContainer"] *,
        .theme-label,
        .chat-disclaimer {{
            color: var(--text-soft) !important;
        }}

        .stButton > button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"] {{
            background: var(--button-secondary-bg) !important;
            color: var(--button-secondary-text) !important;
            border: 1px solid var(--button-border) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }}

        .stButton > button:hover,
        [data-testid="baseButton-secondary"]:hover,
        [data-testid="baseButton-primary"]:hover {{
            background: var(--button-hover) !important;
            color: var(--button-secondary-text) !important;
            border-color: var(--accent) !important;
        }}

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: var(--button-bg) !important;
            color: var(--button-text) !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background: var(--dropzone-bg) !important;
            border: 1px dashed var(--dropzone-border) !important;
            border-radius: 18px !important;
        }}

        [data-testid="stFileUploaderDropzone"] * {{
            color: var(--text-soft) !important;
        }}

        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--button-secondary-bg) !important;
            color: var(--button-secondary-text) !important;
            border: 1px solid var(--button-border) !important;
        }}

        [data-testid="stNumberInputContainer"] input,
        [data-testid="stTextInputRootElement"] input,
        textarea {{
            background: var(--input-surface) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--input-border) !important;
        }}

        input::placeholder, textarea::placeholder {{
            color: var(--placeholder) !important;
            opacity: 1 !important;
        }}

        [data-testid="stNumberInputStepUp"],
        [data-testid="stNumberInputStepDown"] {{
            background: var(--button-secondary-bg) !important;
            color: var(--button-secondary-text) !important;
            border-left: 1px solid var(--input-border) !important;
        }}

        [data-testid="stChatMessage"] {{
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            padding: 0.35rem 0.55rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
        }}

        [data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {{
            background: var(--message-user-bg);
            border-radius: 16px;
            padding: 0.35rem 0.4rem;
        }}

        [data-testid="stChatMessageAvatarAssistant"] + [data-testid="stChatMessageContent"] {{
            background: var(--message-assistant-bg);
            border-radius: 16px;
            padding: 0.35rem 0.4rem;
        }}

        .chat-toolbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin: 0.25rem 0 1rem 0;
            flex-wrap: wrap;
        }}

        .chat-toolbar .status-chip {{
            color: var(--text-soft) !important;
            font-size: 0.92rem;
            background: var(--status-bg);
            border: 1px solid var(--panel-border);
            border-radius: 999px;
            padding: 0.9rem 0.8rem;
            backdrop-filter: blur(8px);
        }}

        div[data-testid="stChatInput"] {{
            position: fixed;
            left: clamp(var(--sidebar-width), 23vw, 21rem);
            right: 1.25rem;
            bottom: calc(var(--bottom-footer-height) + 1.15rem);
            width: auto;
            z-index: 50;
            padding-top: 0;
            background: none;
        }}

        div[data-testid="stChatInput"] > div {{
            background: var(--chat-input-bg);
            border: 1px solid var(--composer-border);
            border-radius: 24px;
            padding: 0.42rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }}

        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] input,
        div[data-testid="stChatInput"] button {{
            color: var(--text-main) !important;
        }}

        .bottom-support {{
            position: fixed;
            left: clamp(var(--sidebar-width), 23vw, 21rem);
            right: 1.25rem;
            bottom: 0.75rem;
            z-index: 49;
        }}

        .chat-disclaimer {{
            text-align: center;
            font-size: 0.82rem;
            padding: 0 1rem 0.35rem;
        }}

        .awesome-footer {{
            background: var(--footer-bg);
            color: #f8fafc !important;
            text-align: center;
            padding: 0.85rem 1rem;
            font-size: 0.92rem;
            border-radius: 18px;
            border: 1px solid var(--panel-border);
            box-shadow: var(--shadow);
        }}

        .awesome-footer a {{
            color: var(--footer-link) !important;
            text-decoration: none;
        }}

        .theme-label {{
            font-size: 0.82rem;
            text-align: center;
            padding-top: 0.2rem;
        }}

        @media (max-width: 768px) {{
            [data-testid="stAppViewContainer"] {{
                padding-bottom: 12rem;
            }}

            .block-container {{
                padding-left: 0.85rem;
                padding-right: 0.85rem;
                padding-bottom: 12rem;
            }}

            div[data-testid="stChatInput"] {{
                left: 0.85rem;
                right: 0.85rem;
                bottom: 5.9rem;
            }}

            .bottom-support {{
                left: 0.85rem;
                right: 0.85rem;
                bottom: 0.5rem;
            }}

            [data-testid="stChatMessage"] {{
                border-radius: 14px;
                padding: 0.2rem 0.35rem;
            }}

            .chat-toolbar {{
                align-items: stretch;
            }}

            .chat-toolbar .status-chip {{
                width: 100%;
                text-align: center;
            }}

            .chat-disclaimer {{
                font-size: 0.76rem;
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }}

            .awesome-footer {{
                font-size: 0.84rem;
                padding: 0.75rem 0.85rem;
            }}
        }}
        </style>
        <div data-theme-mode="{theme}"></div>
        """


def _build_footer_markup() -> str:
    current_year = datetime.now().year
    footer_text = html.escape(f"AwesomeBot (c) {current_year} | Made with care by ")
    return (
        '<div class="bottom-support">'
        '<div class="chat-disclaimer">AwesomeBot can make mistakes. Double-check important info.</div>'
        f'<div class="awesome-footer">{footer_text} <a href="https://awesomekoder.com" target="_blank" rel="noopener noreferrer">AWESOME KODER</a></div>'
        '</div>'
    )


def _inject_styles():
    st.markdown(_build_app_styles(), unsafe_allow_html=True)


def _render_footer():
    st.markdown(_build_footer_markup(), unsafe_allow_html=True)

def _render_chat_toolbar():
    chunk_count = st.session_state.get("doc_chunks", 0)
    if st.session_state.get("vector_store") is not None:
        status = f"Indexed chunks: {chunk_count}"
    else:
        status = "No PDFs indexed yet"

    current_theme = st.session_state.get(THEME_KEY, "light")
    theme_button_label = "Dark Mode" if current_theme == "light" else "Light Mode"

    left_col, toggle_col, action_col = st.columns([0.56, 0.16, 0.28])
    with left_col:
        st.markdown(
            f"""
            <div class="chat-toolbar">
                <div class="status-chip">{html.escape(status)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_col:
        summary_clicked = st.button("Generate Summary", use_container_width=True)
    return summary_clicked


def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon=":scroll:", layout="wide")
    _init_session_state()
    _inject_styles()
    _render_header()

    pdf_docs, max_chunks, process_clicked = render_sidebar()

    uploaded_names = [doc.name for doc in pdf_docs] if pdf_docs else []
    previous_names = st.session_state.get("last_uploaded_names", [])

    if not pdf_docs and previous_names:
        _reset_document_state()
        st.session_state["last_uploaded_names"] = []
    elif uploaded_names != previous_names:
        if st.session_state.get("doc_fingerprint") is not None:
            _reset_document_state()
        st.session_state["last_uploaded_names"] = uploaded_names

    if process_clicked:
        build_and_cache_vector_store(pdf_docs, max_chunks)

    summary_clicked = _render_chat_toolbar()
    _render_chat_history()

    if summary_clicked:
        result = handle_summary()
        if result["ok"]:
            _stream_assistant_message(
                result["message"],
                source=result.get("source"),
                allow_retry=False,
                prompt=None,
                retry_count=0,
            )
        else:
            _show_status_message(result)

    _handle_retry_request()

    prompt = st.chat_input("Ask a question about your PDFs")
    if prompt is not None:
        _handle_user_prompt(prompt)

    _render_footer()
    _scroll_chat_to_bottom()


if __name__ == "__main__":
    main()
