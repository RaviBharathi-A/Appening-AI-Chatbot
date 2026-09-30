"""Parse the Agentic AI eBook and store page-aware chunks locally."""

import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.vector_store import LocalVectorStore


def run_ingestion(pdf_path: str | Path) -> int:
    """Replace the local vector collection with chunks extracted from ``pdf_path``."""
    path = Path(pdf_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    settings = get_settings()

    documents = PyPDFLoader(str(path)).load()
    if not documents or not any(document.page_content.strip() for document in documents):
        raise ValueError(f"No extractable text was found in PDF: {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["source"] = path.name
        chunk.metadata["page"] = int(chunk.metadata["page"]) + 1

    vector_store = LocalVectorStore(
        path=settings.local_vector_store_path,
        collection=settings.local_vector_store_collection,
    )
    return vector_store.replace_documents(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the Agentic AI eBook PDF into the local vector store."
    )
    parser.add_argument(
        "--pdf-path",
        default=r"D:\Placement\Ebook-Agentic-AI.pdf",
        help="Path to the source PDF (default: D:\\Placement\\Ebook-Agentic-AI.pdf).",
    )
    args = parser.parse_args()
    count = run_ingestion(args.pdf_path)
    print(f"Ingested {count} chunks into the local vector store.")


if __name__ == "__main__":
    main()
