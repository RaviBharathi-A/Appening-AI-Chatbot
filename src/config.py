"""Application settings for local document retrieval."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Validated settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    local_vector_store_path: str = Field(
        default="data/agentic-ai-vectors.sqlite3", min_length=1
    )
    local_vector_store_collection: str = Field(
        default="agentic-ai-ebook", min_length=1
    )
    retrieval_top_k: int = Field(default=8, ge=1, le=20)
    retrieval_score_threshold: float = Field(default=0.08, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and fail clearly when required configuration is missing."""
    return Settings()
