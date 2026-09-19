"""
Database connection and schema management using Supabase SDK.
Handles Supabase Postgres with pgvector extension for RAG operations.
"""

import logging
from typing import Optional
from supabase import Client, create_client
from app.core.config import settings

logger = logging.getLogger(__name__)

class Database:
    """Supabase database operations manager for RAG functionality."""

    def __init__(self) -> None:
        """Initialize Supabase client."""
        self.supabase: Optional[Client] = None
        self._admin_client: Optional[Client] = None

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
        Check if database schema is initialized.
        
        Note: The actual schema setup must be done manually in Supabase.
        Run the SQL script: sql/init_supabase.sql in your Supabase SQL Editor.
        """
        try:
            client = self.get_client(admin=True)
            
            # Check if the table exists and has the required structure
            result = client.table('rag_chunks').select('id').limit(1).execute()
            
            if result.data is not None:
                logger.info("Database schema is properly initialized")
                
                # Test the vector search function
                try:
                    stats_result = client.rpc('get_chunk_stats').execute()
                    if stats_result.data:
                        stats = stats_result.data[0]
                        logger.info(f"Database stats: {stats['total_chunks']} chunks, {stats['unique_sources']} sources")
                except Exception:
                    logger.warning("get_chunk_stats function not available - some features may not work")
                    
            else:
                logger.error("Database schema not initialized!")
                logger.error(
                    "Please run the SQL initialization script:\n"
                    "1. Open your Supabase project dashboard\n"
                    "2. Go to SQL Editor\n"
                    "3. Run the script: sql/init_supabase.sql\n"
                    "4. Restart your application"
                )
                
        except Exception as e:
            logger.error(f"Database schema check failed: {e}")
            logger.error(
                "Please ensure you've run sql/init_supabase.sql in your Supabase dashboard"
            )

    


db = Database()
