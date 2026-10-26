"""
Seed the knowledge base: chunk -> embed -> upsert into rag_documents.

Run directly to seed the default documents without starting the API:
    venv/bin/python -m app.rag.ingest
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.data.default_documents import DEFAULT_DOCUMENTS
from app.rag.vectorstore import SupabaseStore
from app.services.chunker import chunker

logger = logging.getLogger(__name__)


def chunks_to_documents(chunks: List[Dict[str, str]]) -> List[Document]:
    """Turn TextChunker output into LangChain Documents with our metadata."""
    documents = []
    for chunk in chunks:
        doc_id, _, index = chunk["chunk_id"].rpartition("#")
        documents.append(Document(
            id=chunk["chunk_id"],
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "doc_id": doc_id,
                "source": chunk["source"],
                "chunk_index": int(index),
            },
        ))
    return documents


async def seed_documents(
    store: SupabaseStore,
    embeddings: Embeddings,
    documents: Optional[List[Dict[str, str]]] = None,
) -> int:
    """Chunk, embed and store documents. Returns the number of chunks written.

    Re-seeding is safe: chunk ids are primary keys, so existing chunks are
    updated, and chunks a shorter new version no longer has are removed.
    """
    started = time.time()
    if documents is None:
        documents = DEFAULT_DOCUMENTS
        logger.info("Using default documents for seeding")

    chunk_docs = chunks_to_documents(chunker.chunk_documents(documents))
    if not chunk_docs:
        return 0

    vectors = await embeddings.aembed_documents([d.page_content for d in chunk_docs])
    ids = [d.id for d in chunk_docs]
    written = await asyncio.to_thread(store.add_vectors, vectors, chunk_docs, ids)

    ids_by_doc = defaultdict(list)
    for doc in chunk_docs:
        ids_by_doc[doc.metadata["doc_id"]].append(doc.id)
    for doc_id, keep_ids in ids_by_doc.items():
        await asyncio.to_thread(store.delete_stale_chunks, doc_id, keep_ids)

    logger.info(
        "Seeded %d chunks from %d documents in %dms",
        len(written), len(documents), int((time.time() - started) * 1000),
    )
    return len(written)


async def _main() -> None:
    from app.core.database import db
    from app.rag.vectorstore import make_embeddings, make_vectorstore

    logging.basicConfig(level=logging.INFO)
    await db.connect()
    embeddings = make_embeddings()
    count = await seed_documents(make_vectorstore(db.get_client(admin=True), embeddings), embeddings)
    print(f"Seeded {count} chunks")


if __name__ == "__main__":
    asyncio.run(_main())
