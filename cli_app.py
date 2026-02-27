import argparse
from pathlib import Path

from app_core import (
    answer_question,
    build_vector_store,
    extract_text_from_pdfs,
    resolve_api_key,
    split_text,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ask questions over PDF documents.")
    parser.add_argument(
        "--pdf",
        nargs="+",
        required=True,
        help="One or more PDF paths.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the PDFs.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Google API key. Falls back to GOOGLE_API_KEY or GEMINI_API_KEY env vars.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=120,
        help="Limit chunks to control cost and latency.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pdf_paths = [Path(p) for p in args.pdf]
    missing = [str(p) for p in pdf_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing PDFs: {', '.join(missing)}")

    resolve_api_key(args.api_key)

    raw_text, errors = extract_text_from_pdfs(pdf_paths)
    if errors:
        for error in errors:
            print(f"Warning: {error}")
    if not raw_text:
        raise SystemExit("No text could be extracted from the PDFs.")

    chunks = split_text(raw_text)
    if not chunks:
        raise SystemExit("Chunking produced no data. Try different PDFs.")
    chunks = chunks[: args.max_chunks]

    vector_store = build_vector_store(chunks)
    answer = answer_question(vector_store, args.question)
    print(answer)


if __name__ == "__main__":
    main()
