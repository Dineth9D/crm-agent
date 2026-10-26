"""
Vector store for the RAG pipeline.

SupabaseStore is a small LangChain VectorStore over the rag_documents table
(sql/migrations/002_langchain_vectorstore.sql). We keep our own class because
langchain-community, home of the old SupabaseVectorStore, is archived.

The supabase-py client is synchronous, so the store's methods are sync.
ChunkRetriever is the async entry point: it embeds with the async OpenAI
client and runs the store calls in a worker thread, so the event loop is
never blocked.
"""

import asyncio
import logging
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from supabase import Client

from app.core.config import settings

logger = logging.getLogger(__name__)

ScoredDocument = Tuple[Document, float]


def make_embeddings() -> OpenAIEmbeddings:
    """OpenAI embeddings, configured to match the vectors already stored."""
    return OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
        # Send raw strings, as the old EmbeddingService did. With the default
        # (True) LangChain tokenizes first, which needs a tiktoken download.
        check_embedding_ctx_length=False,
    )


class SupabaseStore(VectorStore):
    """LangChain vector store over a Supabase table and its match function."""

    def __init__(
        self,
        client: Client,
        embedding: Embeddings,
        table_name: str = "rag_documents",
        query_name: str = "match_documents",
    ) -> None:
        self._client = client
        self._embedding = embedding
        self.table_name = table_name
        self.query_name = query_name

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ---- writes -------------------------------------------------------------

    def add_vectors(
        self,
        vectors: Sequence[List[float]],
        documents: Sequence[Document],
        ids: Sequence[str],
    ) -> List[str]:
        """Upsert pre-computed vectors. Each id is the primary key (chunk_id)."""
        if not (len(vectors) == len(documents) == len(ids)):
            raise ValueError("vectors, documents and ids must have the same length")
        if not ids:
            return []

        rows = [
            {
                "id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": vector,
            }
            for doc_id, doc, vector in zip(ids, documents, vectors)
        ]
        result = self._client.table(self.table_name).upsert(rows).execute()
        written = [str(row["id"]) for row in (result.data or [])]
        logger.info("Upserted %d chunks into %s", len(written), self.table_name)
        return written

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        if ids is None:
            raise ValueError("SupabaseStore needs explicit ids (the chunk ids)")
        metadatas = metadatas or [{} for _ in texts]
        documents = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return self.add_vectors(self._embedding.embed_documents(texts), documents, ids)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return False
        self._client.table(self.table_name).delete().in_("id", ids).execute()
        return True

    def delete_stale_chunks(self, doc_id: str, keep_ids: Sequence[str]) -> int:
        """Remove chunks of doc_id that are not in keep_ids.

        When an edited document gets shorter, its old trailing chunks
        (e.g. #4 after a re-seed that only makes #1-#3) would otherwise stay
        in the table and keep matching searches.
        """
        query = (
            self._client.table(self.table_name)
            .delete()
            .eq("metadata->>doc_id", doc_id)
        )
        if keep_ids:
            query = query.not_.in_("id", list(keep_ids))
        result = query.execute()
        removed = len(result.data or [])
        if removed:
            logger.info("Removed %d stale chunks of %s", removed, doc_id)
        return removed

    # ---- reads --------------------------------------------------------------

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[ScoredDocument]:
        """Nearest chunks with their cosine similarity (1.0 = identical)."""
        params = {"query_embedding": embedding, "match_count": k, "filter": filter or {}}
        result = self._client.rpc(self.query_name, params).execute()
        return [
            (
                Document(id=row["id"], page_content=row["content"], metadata=row["metadata"] or {}),
                float(row["similarity"]),
            )
            for row in (result.data or [])
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[ScoredDocument]:
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(embedding, k, filter)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        pairs = self.similarity_search_by_vector_with_score(embedding, k, kwargs.get("filter"))
        return [doc for doc, _ in pairs]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # match_documents already returns cosine similarity in [0, 1] for
        # these embeddings, so it is the relevance score as-is.
        return lambda score: score

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        client: Optional[Client] = None,
        **kwargs: Any,
    ) -> "SupabaseStore":
        if client is None:
            raise ValueError("SupabaseStore.from_texts needs client=")
        store = cls(client, embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store


def make_vectorstore(client: Client, embeddings: Embeddings) -> SupabaseStore:
    return SupabaseStore(
        client,
        embeddings,
        table_name=settings.vector_table,
        query_name=settings.vector_query_name,
    )


class ChunkRetriever:
    """Async search over the store, with the MIN_SIMILARITY filter."""

    def __init__(
        self,
        store: SupabaseStore,
        embeddings: Embeddings,
        min_similarity: float = 0.0,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.min_similarity = min_similarity

    async def search_by_vector(self, embedding: List[float], k: int) -> List[ScoredDocument]:
        """Raw results, no threshold. The eval uses this to see every score."""
        return await asyncio.to_thread(
            self.store.similarity_search_by_vector_with_score, embedding, k
        )

    async def search(self, query: str, k: int) -> List[ScoredDocument]:
        """Embed the query, search, and drop matches below min_similarity."""
        embedding = await self.embeddings.aembed_query(query)
        results = await self.search_by_vector(embedding, k)
        kept = [(doc, score) for doc, score in results if score >= self.min_similarity]
        logger.info(
            "Search returned %d chunks, %d at or above %.2f",
            len(results), len(kept), self.min_similarity,
        )
        return kept


def doc_id_of(doc: Document) -> str:
    """Document id of a chunk: metadata doc_id, else the chunk id before '#'."""
    if doc.metadata.get("doc_id"):
        return doc.metadata["doc_id"]
    return chunk_id_of(doc).split("#")[0]


def chunk_id_of(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.id or ""


def prepare_context(results: Sequence[ScoredDocument], max_chunks: int = 4) -> List[Document]:
    """Keep the best chunk of each document, up to max_chunks.

    Same rule as the old RAGService._prepare_context. Results arrive sorted
    by similarity, so the first chunk seen for a document is its best one.
    """
    seen_docs = set()
    context: List[Document] = []
    for doc, _score in results:
        doc_id = doc_id_of(doc)
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        context.append(doc)
        if len(context) >= max_chunks:
            break
    return context
