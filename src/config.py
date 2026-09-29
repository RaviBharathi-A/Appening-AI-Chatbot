"""Application settings and Pinecone index validation."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pinecone import Pinecone

load_dotenv()


class Settings(BaseSettings):
    """Validated settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(min_length=1)
    pinecone_api_key: str = Field(min_length=1)
    pinecone_index_name: str = Field(default="agentic-ai-index", min_length=1)
    pinecone_namespace: str = Field(default="agentic-ai-ebook", min_length=1)
    openai_chat_model: str = Field(default="gpt-4o-mini", min_length=1)
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", min_length=1
    )
    retrieval_top_k: int = Field(default=4, ge=1, le=20)
    retrieval_score_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and fail clearly when required configuration is missing."""
    return Settings()


def validate_pinecone_index(settings: Settings) -> None:
    """Ensure the configured Pinecone index exists and matches the embedding size."""
    client = Pinecone(api_key=settings.pinecone_api_key)
    if not client.has_index(settings.pinecone_index_name):
        raise ValueError(
            f"Pinecone index '{settings.pinecone_index_name}' does not exist. "
            "Create it with dimension 1536 and cosine similarity before use."
        )
    index_description = client.describe_index(settings.pinecone_index_name)
    if index_description.dimension != 1536:
        raise ValueError(
            f"Pinecone index '{settings.pinecone_index_name}' has dimension "
            f"{index_description.dimension}; text-embedding-3-small requires 1536."
        )
