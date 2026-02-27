import hashlib
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Tuple

from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain


DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_CHAT_MODEL = "models/gemini-2.0-flash"


def _load_env():
    load_dotenv()
    # Support a legacy key-only .env file in faiss_index/.env
    legacy_env = Path(__file__).resolve().parent / "faiss_index" / ".env"
    if legacy_env.exists():
        content = legacy_env.read_text(encoding="utf-8").strip()
        if content and "=" not in content:
            os.environ.setdefault("GOOGLE_API_KEY", content)


def resolve_api_key(api_key: str | None = None) -> str:
    _load_env()
    resolved = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not resolved:
        raise ValueError(
            "Missing API key. Set GOOGLE_API_KEY in .env or pass it in the UI/CLI."
        )
    genai.configure(api_key=resolved)
    return resolved


def extract_text_from_pdfs(pdf_docs: Iterable) -> Tuple[str, List[str]]:
    text_parts: List[str] = []
    errors: List[str] = []

    for pdf in pdf_docs:
        try:
            if hasattr(pdf, "read"):
                pdf_bytes = pdf.getvalue() if hasattr(pdf, "getvalue") else pdf.read()
                pdf_stream = BytesIO(pdf_bytes)
                reader = PdfReader(pdf_stream)
            else:
                pdf_path = Path(pdf)
                reader = PdfReader(str(pdf_path))

            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text_parts.append(content)
        except Exception:
            name = getattr(pdf, "name", None) or str(pdf)
            errors.append(f"Skipped a corrupted or unreadable PDF: {name}")

    return "\n".join(text_parts).strip(), errors


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def _embedding_model_candidates() -> List[str]:
    env_override = os.getenv("EMBEDDING_MODEL")
    bare_default = DEFAULT_EMBEDDING_MODEL.replace("models/", "")
    candidates = [env_override, DEFAULT_EMBEDDING_MODEL, bare_default]
    return [c for c in candidates if c]


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _discover_chat_models(limit: int = 10) -> List[str]:
    try:
        available = genai.list_models()
    except Exception:
        return []

    candidates: List[str] = []
    for model in available:
        methods = [
            method.lower()
            for method in getattr(model, "supported_generation_methods", []) or []
        ]
        if "generatecontent" in methods:
            name = getattr(model, "name", "")
            if name:
                candidates.append(name)

    return _dedupe(candidates)[:limit]


def _chat_model_candidates() -> List[str]:
    env_override = os.getenv("CHAT_MODEL")
    bare_default = DEFAULT_CHAT_MODEL.replace("models/", "")
    discovered = _discover_chat_models()
    fallback = [
        DEFAULT_CHAT_MODEL,
        bare_default,
        "models/gemini-2.5-flash",
        "gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "gemini-1.5-pro",
    ]
    return _dedupe([env_override, *discovered, *fallback])


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return "quota" in lowered or "rate limit" in lowered or "429" in lowered


def _is_retryable_chat_model_error(message: str) -> bool:
    lowered = message.lower()
    retryable_signals = [
        "not found",
        "not supported for generatecontent",
        "does not have permission",
        "permission denied",
        "forbidden",
        "unsupported",
        "unavailable",
        "resource exhausted",
        "internal",
        "503",
    ]
    return any(signal in lowered for signal in retryable_signals)


def _embed_with_retry(embeddings, texts: List[str], max_retries: int = 3) -> List[List[float]]:
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            message = str(exc)
            if attempt >= max_retries or not _is_quota_error(message):
                raise
            time.sleep(delay)
            delay *= 2
    return []


def build_vector_store(text_chunks: List[str]):
    if not text_chunks:
        raise ValueError("No text chunks available for embeddings.")

    last_error = None
    for model_name in _embedding_model_candidates():
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
            vectors: List[List[float]] = []
            batch_size = 16
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i : i + batch_size]
                vectors.extend(_embed_with_retry(embeddings, batch))
            text_embedding_pairs = list(zip(text_chunks, vectors))
            return FAISS.from_embeddings(text_embedding_pairs, embeddings)
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "not found" in message or "not supported for embedcontent" in message:
                continue
            raise

    tried = ", ".join(_embedding_model_candidates())
    raise RuntimeError(f"Embedding failed for models: {tried}") from last_error


def build_qa_chain(model_name: str):
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question as detailed as possible from the provided context.\n"
        "If the answer is not in the provided context, just say:\n"
        "\"Answer is not available in the context.\"\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n"
    )
    return create_stuff_documents_chain(model, prompt)


def answer_question(vector_store, question: str, k: int = 5) -> str:
    docs = vector_store.similarity_search(question, k=k)
    tried_models: List[str] = []
    last_error = None
    for model_name in _chat_model_candidates():
        tried_models.append(model_name)
        try:
            chain = build_qa_chain(model_name)
            response = chain.invoke({"context": docs, "question": question})
            if isinstance(response, str):
                return response
            if isinstance(response, dict):
                return response.get("answer") or response.get("output_text") or str(response)
            return str(response)
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if _is_retryable_chat_model_error(message):
                continue
            raise

    tried = ", ".join(tried_models or _chat_model_candidates())
    raise RuntimeError(f"Chat model failed for models: {tried}") from last_error


def fingerprint_uploads(pdf_docs: Iterable) -> str:
    hasher = hashlib.sha256()
    for pdf in pdf_docs:
        name = getattr(pdf, "name", None) or str(pdf)
        hasher.update(name.encode("utf-8", errors="ignore"))
        if hasattr(pdf, "getvalue") or hasattr(pdf, "read"):
            data = pdf.getvalue() if hasattr(pdf, "getvalue") else pdf.read()
        else:
            data = Path(pdf).read_bytes()
        hasher.update(data)
    return hasher.hexdigest()
