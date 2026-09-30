"""Offline tests for first-time and repeated Pinecone ingestion."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.config import Settings
from src.ingestion import run_ingestion


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.pdf_path = Path(self.temp_dir.name) / "ebook.pdf"
        self.pdf_path.write_bytes(b"test PDF placeholder")
        self.settings = Settings(
            local_vector_store_path=str(Path(self.temp_dir.name) / "vectors.sqlite3"),
        )
        self.patches = [
            patch("src.ingestion.get_settings", return_value=self.settings),
            patch("src.ingestion.PyPDFLoader"),
            patch("src.ingestion.LocalVectorStore"),
        ]
        self.mocks = [patcher.start() for patcher in self.patches]
        self.addCleanup(patch.stopall)
        self.settings_mock, self.loader, self.vector_store = self.mocks
        self.loader.return_value.load.return_value = [
            Document(
                page_content="Agentic AI content for ingestion.",
                metadata={"page": 0},
            )
        ]

    def test_ingestion_replaces_local_collection(self) -> None:
        self.vector_store.return_value.replace_documents.return_value = 1

        count = run_ingestion(self.pdf_path)

        self.assertEqual(count, 1)
        self.vector_store.return_value.replace_documents.assert_called_once()
        documents = self.vector_store.return_value.replace_documents.call_args.args[0]
        self.assertEqual(documents[0].metadata["source"], "ebook.pdf")
        self.assertEqual(documents[0].metadata["page"], 1)


if __name__ == "__main__":
    unittest.main()
