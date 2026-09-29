"""Parse the Agentic AI eBook and upsert page-aware chunks to Pinecone."""

import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from src.config import get_settings, validate_pinecone_index


def run_ingestion(pdf_path: str | Path) -> int:
    """Replace this namespace's vectors with chunks extracted from ``pdf_path``."""
    path = Path(pdf_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    settings = get_settings()
    validate_pinecone_index(settings)

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

    client = Pinecone(api_key=settings.pinecone_api_key)
    index = client.Index(settings.pinecone_index_name)
    index.delete(delete_all=True, namespace=settings.pinecone_namespace)

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=settings.pinecone_namespace,
    )
    vector_store.add_documents(chunks)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the Agentic AI eBook PDF into Pinecone."
    )
    parser.add_argument(
        "--pdf-path",
        default=r"D:\Placement\Ebook-Agentic-AI.pdf",
        help="Path to the source PDF (default: D:\\Placement\\Ebook-Agentic-AI.pdf).",
    )
    args = parser.parse_args()
    count = run_ingestion(args.pdf_path)
    print(f"Ingested {count} chunks into the configured Pinecone namespace.")


if __name__ == "__main__":
    main()
