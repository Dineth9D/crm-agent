"""
RAG (Retrieval-Augmented Generation) service implementation.
Orchestrates the complete RAG pipeline: chunk → embed → search → generate.
"""

import logging
from typing import List, Dict, Any, Tuple
import time
import re

from ..core.database import db
from .embedding import embedding_service
from .chat import chat_service
from .chunker import chunker
from ..data.default_documents import DEFAULT_DOCUMENTS

logger = logging.getLogger(__name__)

class RAGService:
    """Main RAG service orchestrating the complete pipeline"""

    def __init__(self):
        self.db = db
        self.embedding_service = embedding_service
        self.chat_service = chat_service
        self.chunker = chunker

    async def seed_documents(self, documents: List[Dict[str, str]] = None) -> int:
        """
        Seed the knowledge base with documents.

        Args:
            documents: Optional list of documents, If None, uses default documents.

        Returns:
            Number of chunks successfully inserted
        """
        start_time = time.time()

        # Use default documents if none provided
        if documents is None:
            documents = DEFAULT_DOCUMENTS
            logger.info('Using default documents for seeding')

        try:
            # Step 1: Chunk documents
            chunks = self.chunker.chunk_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

            # Step 2: Generate embeddings for all chunks
            texts = [chunks['text'] for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(texts)

            # Step 3: Combine chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding

            # Step 4: Store in database
            inserted_count = await self.db.upsert_chunks(chunks)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Seeding completed in {elapsed_ms}ms: {inserted_count} chunks inserted")

            return inserted_count
            
        except Exception as e:
            logger.error(f"Seeding failed: {e}")
            raise