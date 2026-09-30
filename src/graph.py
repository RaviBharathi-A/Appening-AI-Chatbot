"""Grounded LangGraph retrieval, generation, and answer verification."""

from functools import lru_cache
import math
import re
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src.config import Settings, get_settings
from src.vector_store import LocalVectorStore

REFUSAL = "I cannot answer based on the provided document."


class RetrievedChunk(TypedDict):
    text: str
    page: int
    source: str
    relevance: float


class AgentState(TypedDict):
    question: str
    context: list[RetrievedChunk]
    answer: str
    score: float
    grounded: bool


def _page_number(document: Document) -> int:
    page = document.metadata.get("page", 0)
    try:
        return max(1, int(page))
    except (TypeError, ValueError):
        return 1


def build_rag_graph(settings: Settings | None = None):
    """Construct the compiled retrieval-and-grounding graph."""
    config = settings or get_settings()

    vector_store = LocalVectorStore(
        path=config.local_vector_store_path,
        collection=config.local_vector_store_collection,
    )

    def retrieve_node(state: AgentState) -> dict:
        matches = vector_store.similarity_search_with_relevance_scores(
            state["question"], k=config.retrieval_top_k
        )
        context: list[RetrievedChunk] = []
        for document, relevance in matches:
            normalized_relevance = max(0.0, min(1.0, float(relevance)))
            if normalized_relevance < config.retrieval_score_threshold:
                continue
            context.append(
                {
                    "text": document.page_content,
                    "page": _page_number(document),
                    "source": str(document.metadata.get("source", "Agentic AI eBook")),
                    "relevance": normalized_relevance,
                }
            )
        score = (
            sum(chunk["relevance"] for chunk in context) / len(context)
            if context
            else 0.0
        )
        return {"context": context, "score": score}

    def generate_node(state: AgentState) -> dict:
        answer = _extractive_answer(state["question"], state["context"])
        return {"answer": answer}

    def grade_node(state: AgentState) -> dict:
        if state["answer"] == REFUSAL:
            return {"grounded": True, "context": [], "score": 0.0}
        evidence = _normalize_whitespace(
            "\n".join(chunk["text"] for chunk in state["context"])
        )
        answer_paragraphs = [
            _normalize_whitespace(paragraph)
            for paragraph in state["answer"].splitlines()
            if _normalize_whitespace(paragraph)
        ]
        return {
            "grounded": bool(answer_paragraphs)
            and all(paragraph in evidence for paragraph in answer_paragraphs)
        }

    def route_after_retrieval(
        state: AgentState,
    ) -> Literal["generate", "refuse"]:
        return "generate" if state["context"] else "refuse"

    def route_after_grade(state: AgentState) -> Literal["done", "refuse"]:
        return "done" if state["grounded"] else "refuse"

    def refuse_node(_: AgentState) -> dict:
        return {"answer": REFUSAL, "context": [], "score": 0.0}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("refuse", refuse_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve", route_after_retrieval, {"generate": "generate", "refuse": "refuse"}
    )
    workflow.add_edge("generate", "grade")
    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {"done": END, "refuse": "refuse"},
    )
    workflow.add_edge("refuse", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def _configured_graph():
    return build_rag_graph(get_settings())


def _extractive_answer(question: str, context: list[RetrievedChunk]) -> str:
    """Return relevant source paragraphs without adding generated claims."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "according", "be", "by", "can",
        "do", "does", "document", "ebook", "for", "from", "how", "in", "is",
        "it", "of", "on", "or", "outlined", "the", "their", "to", "what",
        "when", "where", "which", "who", "why", "with",
    }
    question_terms = {
        _normalize_term(term)
        for term in re.findall(r"[a-z0-9]+", question.lower())
        if term not in stop_words
    }
    if not question_terms:
        return REFUSAL

    paragraphs = [
        _normalize_whitespace(paragraph)
        for chunk in context
        for paragraph in re.split(r"\n\s*\n", chunk["text"])
        if _normalize_whitespace(paragraph)
    ]
    document_frequency = {
        term: sum(
            term in {_normalize_term(token) for token in re.findall(r"[a-z0-9]+", paragraph.lower())}
            for paragraph in paragraphs
        )
        for term in question_terms
    }
    candidates: list[tuple[float, int, str]] = []
    for position, paragraph in enumerate(paragraphs):
        raw_terms = re.findall(r"[a-z0-9]+", paragraph.lower())
        if len(raw_terms) < 6 or re.match(r"^\d+(?:\.\d+)*\s", paragraph):
            continue
        paragraph_terms = {_normalize_term(term) for term in raw_terms}
        overlap = question_terms & paragraph_terms
        if overlap:
            weighted_overlap = sum(
                math.log((len(paragraphs) + 1) / (document_frequency[term] + 1)) + 1
                for term in overlap
            )
            candidates.append(
                (weighted_overlap / math.sqrt(len(raw_terms)), position, paragraph)
            )
    if not candidates:
        return REFUSAL

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected: list[str] = []
    seen: set[str] = set()
    for _, _, sentence in candidates:
        if sentence not in seen:
            selected.append(sentence)
            seen.add(sentence)
        if len(selected) == 2:
            break
    return "\n".join(selected)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _normalize_term(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 4 and term.endswith("s"):
        return term[:-1]
    return term


def answer_question(query: str, settings: Settings | None = None) -> dict:
    """Run one question through the grounded RAG graph and format its API payload."""
    graph = _configured_graph() if settings is None else build_rag_graph(settings)
    result = graph.invoke(
        {
            "question": query,
            "context": [],
            "answer": "",
            "score": 0.0,
            "grounded": False,
        }
    )
    return {
        "query": query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": [
            f"{chunk['text']} (Source: {chunk['source']}, page {chunk['page']})"
            for chunk in result["context"]
        ],
        "confidence_score": result["score"],
    }
