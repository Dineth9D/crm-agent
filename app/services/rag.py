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

    async def answer_query(self, query: str, top_k: int) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.

        Args:
            query: User's question
            tok_k: Number of chunk to retrieve

        Returns:
            Dictionary with answer, citations, and debug info
        """
        start_time = time.time()

        try:
            # Step 1: Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)

            # Step 2: Vector similarity search
            search_results = await self.db.vector_search(query_embedding, top_k)

            if not search_results:
                return {
                    'text': "I don't have enough information to answer your question. Could you please rephrase or provide more context?",
                    'citations': [],
                    'debug': {
                        'top_doc_ids': [],
                        'latency_ms': int((time.time() - start_time) * 1000)
                    }
                }

            # Step 3: Deduplicate and prepare context
            context_blocks = self._prepare_context(search_results)


    def prepare_context(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare context blocks from search results with simple deduplication.
        
        Args:
            search_results: Raw search results from database

        Returns:
            Processed context blocks
        """
        # Simple deduplication by chunk_id prefix (MMR-lite)
