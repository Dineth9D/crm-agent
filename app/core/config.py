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
        ..., description="Publishable (anon) key, safe for client-side use"
    )
    supabase_service_key: Optional[str] = Field(
        default=None,
        description="Service role key. Bypasses RLS - server-side only.",
    )

    # Database behaviour
    supabase_schema: str = Field(
        default="public", description="Postgres schema PostgREST exposes"
    )
    require_schema: bool = Field(
        default=False,
        description="Fail startup when a required table is missing instead of warning",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()


settings = get_settings()
