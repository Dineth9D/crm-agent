-- Instructions:
-- 1. Open the SQL Editor in your Supabase dashboard
-- 2. Paste this ENTIRE script and click "Run"
-- 3. Seed: curl -X POST http://localhost:8000/seed -H "X-Admin-Key: $ADMIN_API_KEY"
--    (or: venv/bin/python -m app.rag.ingest)
-- 4. Check the counts with the comparison query at the end

-- =============================================================================
-- STEP 1: Table
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    -- chunk_id, doc_id, source, chunk_index
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1536) NOT NULL,  -- text-embedding-3-small
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- STEP 2: Indexes
-- =============================================================================

-- HNSW, not IVFFlat: IVFFlat searches only the nearest list, so on a small
-- table most searches return no rows.
CREATE INDEX IF NOT EXISTS rag_documents_embedding_hnsw
    ON rag_documents USING hnsw (embedding vector_cosine_ops);

-- Serves metadata @> filter and the stale-chunk cleanup by doc_id.
CREATE INDEX IF NOT EXISTS rag_documents_metadata_gin
    ON rag_documents USING gin (metadata jsonb_path_ops);

-- =============================================================================
-- STEP 3: Search function
-- =============================================================================

-- LIMIT sits inside the function, next to ORDER BY distance.
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(1536),
  match_count int DEFAULT 6,
  filter jsonb DEFAULT '{}'
)
RETURNS TABLE (
  id text,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM rag_documents d
  WHERE d.metadata @> filter
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION get_chunk_stats()
RETURNS TABLE (
  total_chunks bigint,
  unique_sources bigint,
  latest_chunk timestamptz
)
LANGUAGE sql STABLE
AS $$
  SELECT
    COUNT(*),
    COUNT(DISTINCT metadata->>'source'),
    MAX(created_at)
  FROM rag_documents;
$$;

-- =============================================================================
-- STEP 4: Security
-- =============================================================================

-- RLS on with no policies: only the secret (service role) key, which the
-- backend uses, can read or write. The old "anonymous read" dev policy on
-- rag_chunks is intentionally not carried over.
ALTER TABLE rag_documents ENABLE ROW LEVEL SECURITY;

REVOKE EXECUTE ON FUNCTION match_documents(vector, int, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION get_chunk_stats() FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION match_documents(vector, int, jsonb) TO service_role;
GRANT EXECUTE ON FUNCTION get_chunk_stats() TO service_role;

-- =============================================================================
-- STEP 5: Verification
-- =============================================================================

SELECT 'rag_documents table created' AS test_result
WHERE EXISTS (
  SELECT 1 FROM information_schema.tables
  WHERE table_schema = 'public' AND table_name = 'rag_documents'
);

SELECT 'match_documents function created' AS test_result
WHERE EXISTS (
  SELECT 1 FROM information_schema.routines
  WHERE routine_schema = 'public' AND routine_name = 'match_documents'
);

SELECT 'HNSW index created' AS test_result
WHERE EXISTS (
  SELECT 1 FROM pg_indexes
  WHERE tablename = 'rag_documents' AND indexname = 'rag_documents_embedding_hnsw'
);

SELECT 'RLS enabled on rag_documents' AS test_result
WHERE EXISTS (
  SELECT 1 FROM pg_class WHERE relname = 'rag_documents' AND relrowsecurity
);
