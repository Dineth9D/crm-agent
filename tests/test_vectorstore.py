"""Offline tests for SupabaseStore, ChunkRetriever, prepare_context and ingest."""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.rag.ingest import chunks_to_documents, seed_documents
from app.rag.vectorstore import ChunkRetriever, SupabaseStore, prepare_context


class FakeResult:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    """Records the chained postgrest calls and returns canned data."""

    def __init__(self, log, data):
        self.log, self.data = log, data

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.append((name, args))
            return self
        return call

    @property
    def not_(self):
        self.log.append(("not_", ()))
        return self

    def execute(self):
        return FakeResult(self.data)


class FakeClient:
    def __init__(self, rpc_rows=None):
        self.log = []
        self.rpc_rows = rpc_rows or []

    def table(self, name):
        self.log.append(("table", (name,)))
        return FakeQuery(self.log, data=[{"id": "x"}])

    def rpc(self, name, params):
        self.log.append(("rpc", (name, params)))
        return FakeQuery(self.log, data=self.rpc_rows)


def doc(chunk_id):
    return Document(id=chunk_id, page_content=chunk_id, metadata={
        "chunk_id": chunk_id, "doc_id": chunk_id.split("#")[0],
    })


def row(chunk_id, similarity):
    return {"id": chunk_id, "content": "text", "metadata": {"chunk_id": chunk_id}, "similarity": similarity}


def test_search_sends_match_count_and_returns_scores():
    client = FakeClient(rpc_rows=[row("a#1", 0.9), row("b#1", 0.4)])
    store = SupabaseStore(client, DeterministicFakeEmbedding(size=8))

    pairs = store.similarity_search_by_vector_with_score([0.0] * 8, k=3)

    name, params = client.log[0][1]
    assert name == "match_documents"
    assert params["match_count"] == 3 and params["filter"] == {}
    assert [(d.id, s) for d, s in pairs] == [("a#1", 0.9), ("b#1", 0.4)]


async def test_retriever_applies_min_similarity():
    client = FakeClient(rpc_rows=[row("a#1", 0.9), row("b#1", 0.4)])
    embeddings = DeterministicFakeEmbedding(size=8)
    retriever = ChunkRetriever(SupabaseStore(client, embeddings), embeddings, min_similarity=0.5)

    assert [d.id for d, _ in await retriever.search("returns?", k=6)] == ["a#1"]
    # search_by_vector is the raw view the eval uses: no threshold.
    assert len(await retriever.search_by_vector([0.0] * 8, k=6)) == 2


def test_prepare_context_keeps_best_chunk_per_document():
    results = [(doc("a#2"), 0.9), (doc("a#1"), 0.8), (doc("b#1"), 0.7), (doc("c#1"), 0.6)]
    assert [d.id for d in prepare_context(results, max_chunks=2)] == ["a#2", "b#1"]
    assert prepare_context([], max_chunks=4) == []


def test_chunks_to_documents_builds_metadata():
    [d] = chunks_to_documents([{"chunk_id": "policy_returns_v1#3", "source": "https://x", "text": "t"}])
    assert d.id == "policy_returns_v1#3"
    assert d.metadata == {
        "chunk_id": "policy_returns_v1#3", "doc_id": "policy_returns_v1",
        "source": "https://x", "chunk_index": 3,
    }


async def test_seed_upserts_by_chunk_id_and_removes_stale_chunks():
    client = FakeClient()
    embeddings = DeterministicFakeEmbedding(size=8)
    docs = [{"chunk_id": "faq_v1", "source": "s", "text": "short text"}]

    await seed_documents(SupabaseStore(client, embeddings), embeddings, docs)

    upsert = next(args for name, args in client.log if name == "upsert")
    assert [r["id"] for r in upsert[0]] == ["faq_v1#1"]
    # The stale-chunk delete is scoped to this document and keeps the new ids.
    assert ("eq", ("metadata->>doc_id", "faq_v1")) in client.log
    assert ("in_", ("id", ["faq_v1#1"])) in client.log
