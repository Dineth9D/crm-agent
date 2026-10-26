"""
Hand-checked examples for the retrieval metrics.

Run with:  venv/bin/python -m eval.test_metrics
(Also works with pytest, if you install it.)
"""

import math

from eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    to_doc_ids,
)

RETRIEVED = ["a", "b", "c", "d"]


def test_to_doc_ids_keeps_best_rank():
    assert to_doc_ids(["a#2", "b#1", "a#1", "c#1"]) == ["a", "b", "c"]


def test_hit_at_k():
    assert hit_at_k(RETRIEVED, {"c"}, k=2) == 0.0
    assert hit_at_k(RETRIEVED, {"c"}, k=3) == 1.0


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, {"a", "d"}, k=2) == 0.5
    assert recall_at_k(RETRIEVED, {"a", "d"}, k=4) == 1.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, {"a", "d"}, k=2) == 0.5
    # Only 4 results exist, so k=10 divides by 4, not 10.
    assert precision_at_k(RETRIEVED, {"a", "d"}, k=10) == 0.5
    assert precision_at_k([], {"a"}, k=3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, {"a"}) == 1.0
    assert reciprocal_rank(RETRIEVED, {"c", "d"}) == 1 / 3
    assert reciprocal_rank(RETRIEVED, {"z"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(RETRIEVED, {"a"}, k=3) == 1.0
    assert ndcg_at_k(RETRIEVED, {"a", "b"}, k=3) == 1.0


def test_ndcg_relevant_at_rank_two():
    # DCG = 1/log2(3). IDCG = 1/log2(2) = 1.
    assert math.isclose(ndcg_at_k(RETRIEVED, {"b"}, k=3), 1 / math.log2(3))


def test_ndcg_two_relevant_one_missed_by_cutoff():
    # Relevant: a (rank 1) and d (rank 4). With k=3, only a counts.
    # DCG = 1. IDCG = 1 + 1/log2(3), because a perfect top 3 holds both.
    expected = 1 / (1 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k(RETRIEVED, {"a", "d"}, k=3), expected)


def test_ndcg_nothing_found_is_zero():
    assert ndcg_at_k(RETRIEVED, {"z"}, k=3) == 0.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__} {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
