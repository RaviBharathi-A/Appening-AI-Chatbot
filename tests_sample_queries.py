"""Run the six assessment questions against a locally running FastAPI service."""

import json
import math
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.graph import REFUSAL

SAMPLE_QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the eBook?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]


def main() -> None:
    base_url = "http://127.0.0.1:8000"
    failures = 0
    for query in SAMPLE_QUERIES:
        request = Request(
            f"{base_url}/chat",
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"ERROR: {query}\n{exc}", file=sys.stderr)
            failures += 1
            continue

        required = {
            "query",
            "final_answer",
            "retrieved_context_chunks",
            "confidence_score",
        }
        valid_shape = (
            required.issubset(payload)
            and payload["query"] == query
            and isinstance(payload["final_answer"], str)
            and isinstance(payload["retrieved_context_chunks"], list)
            and all(
                isinstance(chunk, str)
                for chunk in payload["retrieved_context_chunks"]
            )
            and isinstance(payload["confidence_score"], (int, float))
            and math.isfinite(payload["confidence_score"])
            and 0.0 <= payload["confidence_score"] <= 1.0
        )
        out_of_scope_refused = query == SAMPLE_QUERIES[-1] and (
            payload.get("final_answer") == REFUSAL
            and payload.get("confidence_score") == 0.0
            and payload.get("retrieved_context_chunks") == []
        )
        in_scope_answered = valid_shape and (
            query == SAMPLE_QUERIES[-1]
            or (
                payload["final_answer"] != REFUSAL
                and bool(payload["retrieved_context_chunks"])
                and payload["confidence_score"] > 0
            )
        )
        if not valid_shape or not in_scope_answered or (
            query == SAMPLE_QUERIES[-1] and not out_of_scope_refused
        ):
            failures += 1
            print(f"FAIL: {query}\n{json.dumps(payload, indent=2)}")
        else:
            print(f"PASS: {query}\n{json.dumps(payload, indent=2)}")

    if failures:
        raise SystemExit(f"{failures} sample query check(s) failed.")


if __name__ == "__main__":
    main()
