"""
Database connection and schema management using Supabase SDK.
Handles Supabase Postgres with pgvector extension for RAG operations.
"""

import logging
from typing import Optional
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


db = Database()
