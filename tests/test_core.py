from io import BytesIO

from PyPDF2 import PdfWriter

from app_core import answer_question, extract_text_from_pdfs, fingerprint_uploads, split_text


class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


def make_blank_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_extract_text_from_pdfs_blank():
    data = make_blank_pdf_bytes()
    upload = FakeUpload("blank.pdf", data)
    text, errors = extract_text_from_pdfs([upload])
    assert text == ""
    assert errors == [] or "blank.pdf" in errors[0]


def test_split_text_chunks():
    text = "A" * 2000
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 4


def test_fingerprint_uploads_stable():
    data = make_blank_pdf_bytes()
    upload = FakeUpload("blank.pdf", data)
    first = fingerprint_uploads([upload])
    second = fingerprint_uploads([upload])
    assert first == second


class FakeVectorStore:
    def similarity_search(self, question, k=5):
        return [f"doc for: {question} ({k})"]


def test_answer_question_fallbacks_to_next_model(monkeypatch):
    attempts = []

    def fake_candidates():
        return ["models/bad-one", "models/good-one"]

    def fake_build_chain(model_name):
        attempts.append(model_name)
        if model_name == "models/bad-one":
            raise RuntimeError("Model not found")

        class GoodChain:
            def invoke(self, payload):
                return {"answer": f"ok: {payload['question']}"}

        return GoodChain()

    monkeypatch.setattr("app_core._chat_model_candidates", fake_candidates)
    monkeypatch.setattr("app_core.build_qa_chain", fake_build_chain)

    result = answer_question(FakeVectorStore(), "What is this?")
    assert result == "ok: What is this?"
    assert attempts == ["models/bad-one", "models/good-one"]


def test_answer_question_raises_for_non_retryable_error(monkeypatch):
    def fake_candidates():
        return ["models/only-one"]

    def fake_build_chain(_):
        class BadChain:
            def invoke(self, _payload):
                raise RuntimeError("Invalid request payload")

        return BadChain()

    monkeypatch.setattr("app_core._chat_model_candidates", fake_candidates)
    monkeypatch.setattr("app_core.build_qa_chain", fake_build_chain)

    try:
        answer_question(FakeVectorStore(), "Question?")
    except RuntimeError as exc:
        assert "Invalid request payload" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for non-retryable error")
