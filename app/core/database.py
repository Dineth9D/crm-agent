"""
Database connection and schema management using Supabase SDK.
Handles Supabase Postgres with pgvector extension for RAG operations.
"""

import logging
from typing import Any, Dict, List, Optional
from supabase import Client, create_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class SchemaNotReady(RuntimeError):
    """Raised when the expected database schema is missing or unusable."""


class Database:
    """Supabase database operations manager for RAG functionality."""

    def __init__(self) -> None:
        """Initialize Supabase client."""
        self.supabase: Optional[Client] = None
        self._admin_client: Optional[Client] = None
        self._schema_ready: bool = False

    async def connect(self) -> None:
        """Establish connection to Supabase."""
        try:
            # Regular client with anon key
            self.supabase = create_client(
                settings.supabase_url,
                settings.supabase_publishable_key
            )
            
            # Admin client with service role key
            self._admin_client = create_client(
                settings.supabase_url,
                settings.supabase_secret_key
            )

            logger.info("Supabase client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Supabase connections (cleanup if needed)."""
        # Supabase clients don't need explicit disconnection
        logger.info("Supabase clients cleaned up")

    def get_client(self, admin: bool = True) -> Client:
        """
        Get Supabase client instance.

        Args:
            admin: If True, returns the admin client with service role key.
                   If False, returns the regular client with anon key.
        """

        if not self.supabase or not self._admin_client:
            raise RuntimeError("Supabase client is not connected. Call connect() first.")
        

        return self._admin_client if admin else self.supabase

    async def initialize_schema(self) -> None:
        """
        Verify the database schema is present and usable.

        The schema itself is created manually: run sql/sql/init_supabase.sql in
        the Supabase SQL Editor. This only checks that it was done.

        Raises:
            SchemaNotReady: when the check fails and settings.require_schema is
                True. When it is False the problem is logged as a warning and
                startup continues.
        """
        try:
            client = self.get_client(admin=True)
            client.table("rag_chunks").select("id").limit(1).execute()
        except Exception as e:
            self._schema_ready = False
            self._report_schema_problem(
                f"Required table 'rag_chunks' is not queryable: {e}"
            )
            return

        self._schema_ready = True
        logger.info("Database schema is properly initialized")

        # Optional helper. Its absence degrades reporting but not core search,
        # so it never blocks startup.
        try:
            stats_result = client.rpc("get_chunk_stats").execute()
            if stats_result.data:
                stats = stats_result.data[0]
                logger.info(
                    f"Database stats: {stats['total_chunks']} chunks, "
                    f"{stats['unique_sources']} sources"
                )
        except Exception:
            logger.warning(
                "get_chunk_stats function not available - some features may not work"
            )

    def _report_schema_problem(self, detail: str) -> None:
        """Raise or warn about a schema problem, per settings.require_schema."""
        remediation = (
            f"{detail}\n"
            "Run the SQL initialization script:\n"
            "1. Open your Supabase project dashboard\n"
            "2. Go to SQL Editor\n"
            "3. Run the script: sql/sql/init_supabase.sql\n"
            "4. Restart your application"
        )

        if settings.require_schema:
            logger.error(remediation)
            raise SchemaNotReady(detail)

        logger.warning(
            "%s\nStarting anyway because require_schema is disabled. "
            "RAG endpoints will fail until this is resolved.",
            remediation,
        )

    @property
    def schema_ready(self) -> bool:
        """Whether the last schema check succeeded."""
        return self._schema_ready

    async def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Upsert document chunks into the database using Supabase SDK.
        
        Args:
            chunks: List of chunk dictionaries with keys: chunk_id, source, text, embedding
            
        Returns:
            Number of chunks inserted
        """
        if not chunks:
            return 0
        
        try:
            client = self.get_client(admin=True)
            
            # Prepare data for Supabase
            chunk_data = []
            for chunk in chunks:
                chunk_data.append({
                    'chunk_id': chunk['chunk_id'],
                    'source': chunk['source'],
                    'text': chunk['text'],
                    'embedding': chunk['embedding']  # Supabase handles vector serialization
                })
            
            # Use upsert with on_conflict parameter
            result = client.table('rag_chunks').upsert(
                chunk_data,
                on_conflict='chunk_id'
            ).execute()
            
            inserted_count = len(result.data) if result.data else 0
            logger.info(f"Upserted {inserted_count} chunks to database")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Failed to upsert chunks: {e}")
            raise
    
    async def vector_search(self, query_embedding: List[float], top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Supabase SDK.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with similarity scores
        """
        try:
            client = self.get_client(admin=True)
            
            # Use RPC call for vector similarity search
            # This requires creating a PostgreSQL function in Supabase
            result = client.rpc('match_chunks', {
                'query_embedding': query_embedding,
                'match_count': top_k
            }).execute()
            
            if result.data:
                logger.info(f"Vector search returned {len(result.data)} results")
                return result.data
            else:
                # Fallback: if RPC function doesn't exist, use regular query
                # This won't have vector similarity but will work for basic testing
                logger.info("Using fallback query (match_chunks RPC function not available)")
                result = client.table('rag_chunks').select('*').limit(top_k).execute()
                
                # Add mock similarity scores for fallback
                if result.data:
                    for i, chunk in enumerate(result.data):
                        chunk['similarity'] = 1.0 - (i * 0.1)  # Mock decreasing similarity
                
                return result.data or []
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Supabase connection health."""
        try:
            client = self.get_client(admin=True)
            
            # Simple query to test connection
            result = client.table('rag_chunks').select('id').limit(1).execute()
            
            return True  # If no exception, connection is healthy
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

db = Database()
