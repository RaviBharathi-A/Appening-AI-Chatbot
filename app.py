"""FastAPI interface for the Agentic AI eBook RAG chatbot."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings, validate_pinecone_index
from src.graph import answer_question

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    description="Answers questions using retrieved evidence from the Agentic AI eBook.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    query: str = Field(min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def _check_index() -> None:
    validate_pinecone_index(get_settings())


@app.get("/health")
def health() -> dict[str, str]:
    try:
        _check_index()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG service is not ready: {exc}") from exc
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        _check_index()
        result = answer_question(request.query)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to answer query: {exc}") from exc
    return QueryResponse(**result)
