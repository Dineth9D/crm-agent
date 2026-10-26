from functools import lru_cache
from typing import Literal

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
    # Chunks below this cosine similarity count as "no data". 0.0 turns it off.
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0, env="MIN_SIMILARITY")
    chunk_size: int = Field(default=400)  # Approximate tokens
    chunk_overlap: int = Field(default=60)  # 15% overlap
    temperature: float = Field(default=0.1)
    embedding_dimensions: int = Field(default=1536)  # text-embedding-3-small dimensions
    max_tokens: int = Field(default=1200)
    # Chunks the answer model sees, at most one per document.
    max_context_chunks: int = Field(default=4, ge=1)

    # Vector store (sql/migrations/002_langchain_vectorstore.sql)
    vector_table: str = Field(default="rag_documents")
    vector_query_name: str = Field(default="match_documents")

    # LangGraph behaviour
    # Searches per question, counting the first one. 2 = one rewrite and retry.
    max_retrieval_attempts: int = Field(default=2, ge=1, le=5)
    # Earlier messages (user + assistant) sent with each turn.
    history_max_messages: int = Field(default=10, ge=0)
    # "memory" loses history on restart. "postgres" needs supabase_db_url.
    checkpointer: Literal["memory", "postgres"] = Field(default="memory")
    supabase_db_url: str = Field(
        default="", description="Postgres URL (session pooler) for the postgres checkpointer"
    )

    # Admin access
    admin_api_key: str = Field(
        default="",
        description=(
            "Shared secret required in the X-Admin-Key header for admin-only "
            "endpoints such as POST /seed. When empty, those endpoints are disabled."
        ),
    )

    # Application Configuration
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_embed_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBED_MODEL")
    openai_chat_model: str = Field(default="gpt-4o", env="OPENAI_CHAT_MODEL")
    # Model for query rewriting and grading. Empty means use openai_chat_model.
    openai_utility_model: str = Field(default="")

    # Anthropic Configuration
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_chat_model: str = Field(default="claude-sonnet-5-5", env="ANTHROPIC_CHAT_MODEL")
    anthropic_utility_model: str = Field(default="claude-haiku-5-5")
    # Current Claude models always think, and thinking counts toward max_tokens,
    # so this needs more room than the OpenAI max_tokens.
    anthropic_max_tokens: int = Field(default=4096)

    # AI Provider for chat. Embeddings always use OpenAI.
    ai_provider: Literal["openai", "anthropic"] = Field(default="openai", env="AI_PROVIDER")

    # Size limits for request validation
    max_chunk_id_length: int = Field(default=128)
    max_source_length: int = Field(default=2048)
    max_doc_text_length: int = Field(default=50_000)
    max_docs_per_seed: int = Field(default=100)
    max_query_length: int = Field(default=2000)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()


settings = get_settings()
