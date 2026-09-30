"""Offline tests for local TF-IDF persistence and cosine retrieval."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from src.vector_store import LocalVectorStore


class LocalVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "vectors.sqlite3"
        self.store = LocalVectorStore(self.path, "ebook")

    def test_persists_and_ranks_documents_by_tfidf_cosine_similarity(self) -> None:
        self.store.replace_documents(
            [
                Document(
                    page_content="Agentic systems plan workflows.",
                    metadata={"page": 1},
                ),
                Document(
                    page_content="Agentic systems use tools.",
                    metadata={"page": 2},
                ),
                Document(page_content="Tomatoes grow outdoors.", metadata={"page": 3}),
            ]
        )

        reopened_store = LocalVectorStore(self.path, "ebook")
        matches = reopened_store.similarity_search_with_relevance_scores(
            "Agentic systems plan", k=3
        )

        self.assertEqual(
            matches[0][0].page_content,
            "Agentic systems plan workflows.",
        )
        self.assertGreater(matches[0][1], matches[1][1])
        self.assertEqual(matches[0][0].metadata["page"], 1)
        self.assertEqual(reopened_store.count(), 3)
        self.assertTrue(0.0 <= matches[0][1] <= 1.0)

    def test_replacing_collection_removes_old_rows(self) -> None:
        self.store.replace_documents([Document(page_content="First document.")])

        self.store.replace_documents([Document(page_content="Second document.")])

        self.assertEqual(self.store.count(), 1)
        matches = self.store.similarity_search_with_relevance_scores("Second")
        self.assertEqual(matches[0][0].page_content, "Second document.")

    def test_rejects_empty_replacement_before_deleting_existing_rows(self) -> None:
        self.store.replace_documents([Document(page_content="Existing document.")])

        with self.assertRaises(ValueError):
            self.store.replace_documents([])

        self.assertEqual(self.store.count(), 1)

    def test_unmatched_query_has_no_relevance(self) -> None:
        self.store.replace_documents([Document(page_content="Agentic systems plan.")])

        matches = self.store.similarity_search_with_relevance_scores("volcanoes")

        self.assertEqual(matches, [])

    def test_normalizes_common_plural_forms(self) -> None:
        self.store.replace_documents(
            [Document(page_content="Industries adopt different systems.")]
        )

        matches = self.store.similarity_search_with_relevance_scores("industry system")

        self.assertEqual(len(matches), 1)
        self.assertGreater(matches[0][1], 0)

    def test_migrates_legacy_openai_embedding_schema(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("DROP TABLE vectors")
            connection.execute(
                """
                CREATE TABLE vectors (
                    collection TEXT NOT NULL,
                    vector_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    PRIMARY KEY (collection, vector_id)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO vectors
                    (collection, vector_id, content, metadata, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "ebook",
                    "old-id",
                    "Agentic AI systems plan workflows.",
                    json.dumps({"page": 4}),
                    json.dumps([0.1, 0.2, 0.3]),
                ),
            )
            connection.commit()
        finally:
            connection.close()

        migrated_store = LocalVectorStore(self.path, "ebook")
        matches = migrated_store.similarity_search_with_relevance_scores(
            "Agentic systems workflows"
        )

        self.assertEqual(migrated_store.count(), 1)
        self.assertEqual(matches[0][0].metadata["page"], 4)
        self.assertEqual(
            matches[0][0].page_content,
            "Agentic AI systems plan workflows.",
        )


if __name__ == "__main__":
    unittest.main()
