from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Supabase configuration
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_publishable_key: str = Field(
        ..., description="Publishable key, safe for client-side use"
    )
    supabase_secret_key: str = Field(
        ...,description="Secret key. Bypasses RLS - server-side only.",
    )

    # Database behaviour
    supabase_schema: str = Field(
        default="public", description="Postgres schema PostgREST exposes"
    )
    require_schema: bool = Field(
        default=True,
        description="Abort startup when the expected tables are missing, instead of warning",
    )

    # RAG Configuration
    default_top_k: int = Field(default=6)
    chunk_size: int = Field(default=400)  # Approximate tokens
    chunk_overlap: int = Field(default=60)  # 15% overlap
    temperature: float = Field(default=0.1)
    embedding_dimensions: int = Field(default=1536)  # text-embedding-3-small dimensions

    # Application Configuration
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()


settings = get_settings()
