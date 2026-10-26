"""
Retrieval evaluation for the RAG pipeline.

Scores two stages for every question in the dataset:
    search:  raw results from db.vector_search (top_k chunks)
    context: what the LLM actually sees, after rag_service._prepare_context

Usage:
    venv/bin/python -m eval.run_retrieval
    venv/bin/python -m eval.run_retrieval --top-k 6 --dataset eval/datasets/retrieval.jsonl

The knowledge base must be seeded first (POST /seed).
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings
from app.core.database import db
from app.services.embedding import embedding_service
from app.services.rag import rag_service
from eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    to_doc_ids,
)

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "datasets" / "retrieval.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
K_VALUES = (1, 3, 5)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load test cases from a JSONL file, one case per line."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_case(search_docs: List[str], context_docs: List[str], relevant: set) -> Dict[str, float]:
    """Compute all retrieval metrics for one in-scope question."""
    scores: Dict[str, float] = {}
    for k in K_VALUES:
        scores[f"hit@{k}"] = hit_at_k(search_docs, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(search_docs, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(search_docs, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(search_docs, relevant, k)
    scores["mrr"] = reciprocal_rank(search_docs, relevant)

    # The context has no k: it is the full list the LLM receives.
    scores["context_hit"] = hit_at_k(context_docs, relevant, len(context_docs))
    scores["context_recall"] = recall_at_k(context_docs, relevant, len(context_docs))
    return scores


def mean_scores(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average every metric over the given scored rows."""
    if not rows:
        return {}
    keys = rows[0]["scores"].keys()
    return {key: statistics.mean(row["scores"][key] for row in rows) for key in keys}


async def run(dataset_path: Path, top_k: int) -> Dict[str, Any]:
    cases = load_dataset(dataset_path)
    await db.connect()

    # One batched embedding call for all questions: faster and cheaper.
    started = time.time()
    query_embeddings = await embedding_service.embed_texts([c["question"] for c in cases])

    rows: List[Dict[str, Any]] = []
    for case, embedding in zip(cases, query_embeddings):
        search_started = time.time()
        results = await db.vector_search(embedding, top_k)
        search_ms = (time.time() - search_started) * 1000

        context = rag_service._prepare_context(results) if results else []
        search_docs = to_doc_ids([r["chunk_id"] for r in results])
        context_docs = to_doc_ids([c["chunk_id"] for c in context])
        relevant = set(case["relevant_doc_ids"])

        rows.append({
            "id": case["id"],
            "question": case["question"],
            "tag": case.get("tag", ""),
            "relevant": sorted(relevant),
            "search_docs": search_docs,
            "context_docs": context_docs,
            "top1_similarity": results[0].get("similarity") if results else None,
            "num_results": len(results),
            "search_ms": round(search_ms, 1),
            "scores": score_case(search_docs, context_docs, relevant) if relevant else None,
        })

    await db.disconnect()

    in_scope = [r for r in rows if r["scores"] is not None]
    by_tag: Dict[str, Dict[str, float]] = {}
    for tag in sorted({r["tag"] for r in in_scope}):
        by_tag[tag] = mean_scores([r for r in in_scope if r["tag"] == tag])

    return {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "dataset": str(dataset_path),
            "top_k": top_k,
            "embed_model": settings.openai_embed_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "total_seconds": round(time.time() - started, 2),
        "summary": mean_scores(in_scope),
        "by_tag": by_tag,
        "cases": rows,
    }


def print_report(report: Dict[str, Any]) -> None:
    cfg = report["config"]
    cases = report["cases"]
    in_scope = [c for c in cases if c["scores"] is not None]
    out_scope = [c for c in cases if c["scores"] is None]

    print(f"\nRetrieval eval  |  top_k={cfg['top_k']}  embed={cfg['embed_model']}  "
          f"chunk_size={cfg['chunk_size']}  overlap={cfg['chunk_overlap']}")
    print(f"{len(in_scope)} in-scope questions, {len(out_scope)} out-of-scope, "
          f"{report['total_seconds']}s total\n")

    if all(c["num_results"] == 0 for c in cases):
        print("WARNING: every search returned nothing. Is the knowledge base seeded?\n")

    summary = report["summary"]
    print(f"{'metric':<12}" + "".join(f"{'@' + str(k):>9}" for k in K_VALUES))
    for name in ("hit", "recall", "precision", "ndcg"):
        print(f"{name:<12}" + "".join(f"{summary[f'{name}@{k}']:>9.3f}" for k in K_VALUES))
    print(f"\nMRR              {summary['mrr']:.3f}")
    print(f"context hit      {summary['context_hit']:.3f}   (LLM sees a relevant doc)")
    print(f"context recall   {summary['context_recall']:.3f}   (LLM sees all relevant docs)")

    print(f"\n{'tag':<14}{'n':>4}{'hit@1':>9}{'recall@3':>10}{'mrr':>8}")
    for tag, scores in report["by_tag"].items():
        n = sum(1 for c in in_scope if c["tag"] == tag)
        print(f"{tag:<14}{n:>4}{scores['hit@1']:>9.3f}{scores['recall@3']:>10.3f}{scores['mrr']:>8.3f}")

    # Similarity gap: can a threshold tell in-scope from out-of-scope questions?
    in_sims = [c["top1_similarity"] for c in in_scope if c["top1_similarity"] is not None]
    out_sims = [c["top1_similarity"] for c in out_scope if c["top1_similarity"] is not None]
    if in_sims and out_sims:
        print(f"\nTop-1 similarity  in-scope min={min(in_sims):.3f}  mean={statistics.mean(in_sims):.3f}")
        print(f"                  out-of-scope max={max(out_sims):.3f}  mean={statistics.mean(out_sims):.3f}")

    misses = [c for c in in_scope if c["scores"]["hit@1"] == 0.0]
    if misses:
        print(f"\nMissed at rank 1 ({len(misses)}):")
        for c in misses:
            print(f"  {c['id']}  {c['question']}")
            print(f"         expected={c['relevant']}  got={c['search_docs'][:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval accuracy.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    args = parser.parse_args()

    # Keep app INFO logs out of the report.
    logging.basicConfig(level=logging.WARNING)

    report = asyncio.run(run(args.dataset, args.top_k))
    print_report(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {out_path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
