"""Offline tests for graph retrieval gating and extractive grounding."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.config import Settings
from src.graph import REFUSAL, build_rag_graph


def initial_state(question: str = "What can Agentic AI systems do?") -> dict:
    return {
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0,
        "grounded": False,
    }


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store_patch = patch("src.graph.LocalVectorStore")
        self.store = self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        self.settings = Settings(
            local_vector_store_path=str(Path(self.temp_dir.name) / "vectors.sqlite3")
        )

    def test_retrieval_returns_source_sentence_and_grades_it_as_grounded(self) -> None:
        self.store.return_value.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Agentic AI systems can plan and use tools.",
                    metadata={"page": 2, "source": "ebook.pdf"},
                ),
                0.8,
            )
        ]

        result = build_rag_graph(self.settings).invoke(initial_state())

        self.assertEqual(result["answer"], "Agentic AI systems can plan and use tools.")
        self.assertEqual(result["context"][0]["page"], 2)
        self.assertEqual(result["score"], 0.8)
        self.assertTrue(result["grounded"])

    def test_weak_retrieval_refuses(self) -> None:
        self.store.return_value.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="Unrelated excerpt.", metadata={}), 0.02)
        ]

        result = build_rag_graph(self.settings).invoke(initial_state())

        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["context"], [])
        self.assertEqual(result["score"], 0.0)

    def test_answer_not_found_verbatim_in_evidence_is_refused(self) -> None:
        self.store.return_value.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Agentic AI systems plan tasks.",
                    metadata={"page": 1},
                ),
                0.9,
            )
        ]
        with patch(
            "src.graph._extractive_answer",
            return_value="Unsupported claim.",
        ):
            result = build_rag_graph(self.settings).invoke(initial_state())

        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["context"], [])


if __name__ == "__main__":
    unittest.main()
