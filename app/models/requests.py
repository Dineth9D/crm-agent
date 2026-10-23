"""
Request models for the RAG API endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from ..core.config import get_settings

settings = get_settings()


class DocumentChunk(BaseModel):
    """Individual document chunk for seeding."""
    chunk_id: str = Field(
        ..., min_length=1, max_length=settings.max_chunk_id_length,
        description="Unique identifier for the chunk"
    )
    source: str = Field(
        ..., max_length=settings.max_source_length,
        description="Source URL or identifier"
    )
    text: str = Field(
        ..., max_length=settings.max_doc_text_length,
        description="Text content of the chunk"
    )


class SeedRequest(BaseModel):
    """Request model for the /seed endpoint."""
    docs: Optional[List[DocumentChunk]] = Field(
        None,
        max_length=settings.max_docs_per_seed,
        description="Optional list of documents to seed. If omitted, uses default documents."
    )


class AnswerRequest(BaseModel):
    """Request model for the /answer endpoint."""
    query: str = Field(
        ..., min_length=1, max_length=settings.max_query_length,
        description="User question to answer"
    )
    top_k: Optional[int] = Field(
        6, 
        ge=1, 
        le=20, 
        description="Number of chunks to retrieve for context"
    )