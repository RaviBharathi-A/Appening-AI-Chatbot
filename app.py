"""FastAPI interface for the Agentic AI eBook RAG chatbot."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.graph import answer_question
from src.vector_store import LocalVectorStore

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


@app.get("/health")
def health() -> dict[str, str]:
    try:
        settings = get_settings()
        vector_store = LocalVectorStore(
            path=settings.local_vector_store_path,
            collection=settings.local_vector_store_collection,
        )
        if vector_store.count() == 0:
            raise ValueError(
                "No document chunks found. Run the PDF ingestion command first."
            )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"RAG service is not ready: {exc}"
        ) from exc
    return {"status": "ok", "vector_store": "local SQLite"}


@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        result = answer_question(request.query)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to answer query: {exc}") from exc
    return QueryResponse(**result)
