"""Offline tests for the FastAPI request and response contracts."""

import unittest

from pydantic import ValidationError

from app import QueryRequest, QueryResponse
from src.graph import REFUSAL


class ApiContractTests(unittest.TestCase):
    def test_response_has_assignment_fields_and_bounded_score(self) -> None:
        response = QueryResponse(
            query="What is Agentic AI?",
            final_answer="Answer from the source.",
            retrieved_context_chunks=["Evidence (page 2)."],
            confidence_score=0.8,
        )
        self.assertEqual(
            set(response.model_dump()),
            {
                "query",
                "final_answer",
                "retrieved_context_chunks",
                "confidence_score",
            },
        )

    def test_response_rejects_score_outside_unit_interval(self) -> None:
        with self.assertRaises(ValidationError):
            QueryResponse(
                query="Question",
                final_answer=REFUSAL,
                retrieved_context_chunks=[],
                confidence_score=1.1,
            )

    def test_request_rejects_blank_question(self) -> None:
        with self.assertRaises(ValidationError):
            QueryRequest(query="  ")


if __name__ == "__main__":
    unittest.main()
