"""Offline tests for graph retrieval gating and grounded response shape."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.config import Settings
from src.graph import REFUSAL, GroundingGrade, build_rag_graph


def initial_state() -> dict:
    return {
        "question": "What does the eBook say?",
        "context": [],
        "answer": "",
        "score": 0.0,
        "grounded": False,
        "attempts": 0,
    }


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        patches = (
            patch("src.graph.validate_pinecone_index"),
            patch("src.graph.OpenAIEmbeddings"),
            patch("src.graph.PineconeVectorStore"),
            patch("src.graph.ChatOpenAI"),
        )
        self.mocks = [patcher.start() for patcher in patches]
        self.addCleanup(patch.stopall)
        self.embedding, self.store, self.chat = self.mocks[1:]
        self.config = Settings(openai_api_key="test", pinecone_api_key="test")

    def test_retrieval_generates_and_grades_supported_answer(self) -> None:
        self.store.return_value.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Agentic AI systems can plan and use tools.",
                    metadata={"page": 2, "source": "ebook.pdf"},
                ),
                0.8,
            )
        ]
        self.chat.return_value.invoke.return_value = AIMessage(
            content="The eBook describes systems that can plan and use tools."
        )
        self.chat.return_value.with_structured_output.return_value.invoke.return_value = (
            GroundingGrade(grounded=True, explanation="Supported by the excerpt.")
        )

        result = build_rag_graph(self.config).invoke(initial_state())

        self.assertEqual(
            result["answer"],
            "The eBook describes systems that can plan and use tools.",
        )
        self.assertEqual(result["context"][0]["page"], 2)
        self.assertEqual(result["score"], 0.8)
        self.assertTrue(result["grounded"])

    def test_weak_retrieval_refuses_without_calling_language_model(self) -> None:
        self.store.return_value.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="Unrelated excerpt.", metadata={}), 0.2)
        ]

        result = build_rag_graph(self.config).invoke(initial_state())

        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["context"], [])
        self.assertEqual(result["score"], 0.0)
        self.chat.return_value.invoke.assert_not_called()

    def test_ungrounded_answer_is_retried_once_then_refused(self) -> None:
        self.store.return_value.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="Source supports a limited claim.", metadata={}), 0.9)
        ]
        self.chat.return_value.invoke.side_effect = [
            AIMessage(content="Unsupported claim."),
            AIMessage(content="Still unsupported."),
        ]
        self.chat.return_value.with_structured_output.return_value.invoke.return_value = (
            GroundingGrade(grounded=False, explanation="Not supported.")
        )

        result = build_rag_graph(self.config).invoke(initial_state())

        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
