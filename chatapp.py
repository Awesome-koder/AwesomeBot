import logging
import re

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


def show_user_error(message: str):
    st.error(message)


def _ensure_api_key() -> bool:
    try:
        resolve_api_key(None)
        return True
    except ValueError:
        logger.exception("API key resolution failed")
        show_user_error("Configuration is incomplete. Please contact support.")
        return False


def _is_not_in_context(answer_text: str) -> bool:
    return "answer is not available in the context" in (answer_text or "").lower()


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


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
        "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and",
        "or", "in", "on", "for", "with", "at", "by", "from", "what", "which",
        "who", "when", "where", "why", "how", "does", "do", "did", "about",
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


class SummaryVectorStore:
    def __init__(self, summary_text: str):
        self.summary_text = summary_text

    def similarity_search(self, _query: str, k: int = 1):
        if not self.summary_text.strip():
            return []
        return [Document(page_content=self.summary_text)]


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

    process_clicked = st.sidebar.button("Process PDFs", type="primary")

    return pdf_docs, max_chunks, process_clicked


def build_and_cache_vector_store(pdf_docs, max_chunks):
    if not pdf_docs:
        show_user_error("Please upload at least one PDF file to continue.")
        return

    if not _ensure_api_key():
        return

    try:
        with st.spinner("Extracting text..."):
            raw_text, file_errors = extract_text_from_pdfs(pdf_docs)
    except Exception:
        logger.exception("PDF text extraction failed")
        show_user_error("We could not read the uploaded files right now. Please try again.")
        return

    if file_errors:
        st.warning("Some files could not be processed and were skipped.")
        for error in file_errors:
            logger.warning(error)

    if not raw_text:
        show_user_error("No readable text was found in the uploaded PDF files.")
        return

    try:
        with st.spinner("Chunking text..."):
            chunks = split_text(raw_text)
    except Exception:
        logger.exception("Text chunking failed")
        show_user_error("We could not prepare your document for search. Please try again.")
        return

    if not chunks:
        show_user_error("Document processing produced no usable content. Try a different PDF.")
        return

    try:
        chunks = chunks[:max_chunks]
        fingerprint = fingerprint_uploads(pdf_docs)
    except Exception:
        logger.exception("Failed while preparing document metadata")
        show_user_error("We hit a processing issue. Please retry in a moment.")
        return

    with st.spinner("Building vector store..."):
        try:
            vector_store = build_vector_store(chunks)
        except Exception:
            logger.exception("Vector store build failed")
            show_user_error(
                "We could not index your document right now. Please retry shortly."
            )
            return

    st.session_state["vector_store"] = vector_store
    st.session_state["doc_fingerprint"] = fingerprint
    st.session_state["doc_chunks"] = len(chunks)
    st.success("PDFs processed. You can ask questions now.")


def handle_question(question):
    if "vector_store" not in st.session_state:
        st.info("Process PDFs before asking questions.")
        return

    if len(question.strip()) < 4:
        st.warning("Question is too short.")
        return

    answer = None
    source = "index"
    summary_text = (st.session_state.get("latest_summary") or "").strip()

    if summary_text:
        direct_summary_answer = _answer_from_summary_text(question, summary_text)
        if direct_summary_answer and not _is_low_quality_answer(direct_summary_answer, question):
            answer = direct_summary_answer
            source = "summary"
            logger.info("Answered directly from summary text.")

    if summary_text and answer is None:
        if not _ensure_api_key():
            return
        try:
            with st.spinner("Searching summary..."):
                summary_store = SummaryVectorStore(summary_text)
                summary_answer = answer_question(summary_store, question, k=1)
            polished_summary_answer = _polish_summary_answer(summary_answer)
            if (
                not _is_not_in_context(polished_summary_answer)
                and not _is_low_quality_answer(polished_summary_answer, question)
            ):
                answer = polished_summary_answer
                source = "summary"
            else:
                logger.info("Summary answer was weak or missing, falling back to index chunks.")
        except Exception:
            logger.exception("Summary-first lookup failed, falling back to index chunks")

    if answer is None:
        if not _ensure_api_key():
            return
        try:
            with st.spinner("Searching indexed content..."):
                answer = answer_question(st.session_state["vector_store"], question)
        except Exception:
            logger.exception("Question answering failed")
            show_user_error(
                "We could not generate an answer right now. Please try again in a moment."
            )
            return

    st.markdown("**Answer**")
    st.write(answer)
    if source == "summary":
        st.caption("Source: document summary")
    else:
        st.caption("Source: indexed chunks")


def handle_summary():
    if "vector_store" not in st.session_state:
        st.info("Process PDFs before generating a summary.")
        return

    if not _ensure_api_key():
        return

    summary_prompt = (
        "Generate a clear summary of the uploaded PDF content. "
        "Use concise sections with: Overview, Key Points, and Important Details."
    )
    try:
        with st.spinner("Generating summary..."):
            summary = answer_question(st.session_state["vector_store"], summary_prompt)
    except Exception:
        logger.exception("Summary generation failed; attempting local fallback summary")
        summary = build_local_summary(st.session_state["vector_store"])
        if not summary:
            show_user_error(
                "We could not generate a summary right now. Please try again in a moment."
            )
            return
        st.warning("Showing a quick local summary of the uploaded doc.")

    st.session_state["latest_summary"] = summary


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


def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon=":scroll:", layout="wide")
    st.title("PDF Reader AI 🤖")
    st.markdown("""
<style>

/* Target sidebar collapse button */
button[kind="header"] {
    animation: pulseScale 1.8s ease-in-out infinite;
    transition: transform 0.3s ease;
}

/* Smooth scaling animation */
@keyframes pulseScale {
    0% { transform: scale(1); }
    50% { transform: scale(1.18); }
    100% { transform: scale(1); }
}

/* Optional: make animation only visible on mobile */
@media (min-width: 768px) {
    button[kind="header"] {
        animation: none;
    }
}

</style>
""", unsafe_allow_html=True)
    st.markdown("""
<style>
    st.markdown("""
<script>
function openSidebarOnMobile() {
    if (window.innerWidth <= 768) {
        setTimeout(function() {
            const btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
            if (btn) {
                btn.click();
            }
        }, 1000);
    }
}

window.addEventListener("load", openSidebarOnMobile);
</script>
""", unsafe_allow_html=True)
    st.caption("Upload PDFs, build a searchable index, and ask questions with citations from your files.")

    pdf_docs, max_chunks, process_clicked = render_sidebar()

    if process_clicked:
        build_and_cache_vector_store(pdf_docs, max_chunks)

    if "vector_store" in st.session_state:
        if st.button("Generate Summary", use_container_width=False):
            handle_summary()
        if st.session_state.get("latest_summary"):
            st.markdown("**Document Summary**")
            st.write(st.session_state["latest_summary"])

    st.subheader("Ask a question")
    question = st.text_input("Question")
    if question:
        handle_question(question)

    if "doc_chunks" in st.session_state:
        st.caption(f"Chunks indexed: {st.session_state['doc_chunks']}")
    st.markdown(
        """
        <div style="
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #96a7ab;
            color: white;
            padding: 12px;
            text-align: center;
        ">
            © 2026 | Made with  ❤  by <a href="https://awesomekoder.com">AWESOME KODER</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()





