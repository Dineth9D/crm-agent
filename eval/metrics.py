"""
Retrieval metrics for RAG evaluation.

Every function takes:
    retrieved: ranked list of document IDs (best first, no duplicates)
    relevant:  set of document IDs that are correct for the question

Relevance is binary: a document is either relevant or not.
Callers must skip questions with an empty `relevant` set (out-of-scope).
"""

import math
from typing import List, Set


def to_doc_ids(chunk_ids: List[str]) -> List[str]:
    """
    Turn ranked chunk IDs into ranked document IDs.

    'policy_returns_v1#2' -> 'policy_returns_v1'. A document keeps the rank of
    its best chunk, so later chunks of the same document are dropped.
    """
    doc_ids: List[str] = []
    for chunk_id in chunk_ids:
        doc_id = chunk_id.split("#")[0]
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def hit_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    At least one right document show up
    
    1.0 if at least one relevant document is in the top k, else 0.0.
    """
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    How many right documents show up
    
    Share of all relevant documents that appear in the top k.
    """
    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    How many returned documents are right

    Divides by the number of results actually returned (up to k), not by k.
    The knowledge base is small, so fewer than k documents can come back.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / len(top)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant document. 0.0 if none is found."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k, with binary relevance.

    DCG  = sum over the top k of rel_i / log2(i + 1), where i is the 1-based rank
    IDCG = the DCG of a perfect ranking (all relevant documents first)
    nDCG = DCG / IDCG, a value from 0.0 to 1.0
    """
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, doc in enumerate(retrieved[:k], start=1) if doc in relevant)
    idcg = sum((1.0 / math.log2(rank + 1)) for rank in range(1, min(k, len(relevant)) + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
