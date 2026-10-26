
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from emplay_assignment.search.hybrid_search import (
    bm25_search,
    reciprocal_rank_fusion,
)
from emplay_assignment.search.semantic_search import semantic_search


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "evaluation_results.json"

K = 5
CANDIDATE_K = 10


def normalize(value: Any) -> str:
    """Normalize text for reliable source-passage comparison."""
    return re.sub(r"\s+", " ", str(value or "")).casefold().strip()


def load_questions() -> list[dict[str, Any]]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {QUESTIONS_FILE}"
        )

    questions = []

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                question = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            for key in ("id", "query", "bid_id", "expected"):
                if key not in question:
                    raise ValueError(
                        f"Missing '{key}' on evaluation line {line_number}"
                    )

            questions.append(question)

    if len(questions) < 15:
        raise ValueError(
            "Assignment requires at least 15 evaluation questions."
        )

    ids = [question["id"] for question in questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation question IDs found.")

    return questions


def is_expected_passage(
    result: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    """Check whether a result matches the expected source passage."""
    actual_filename = normalize(result.get("file_name"))
    expected_filename = normalize(expected.get("file_contains"))

    if not expected_filename or expected_filename not in actual_filename:
        return False

    # PDF page numbers returned by Pinecone may be floats such as 1.0.
    if "page_number" in expected:
        actual_page = result.get("page_number")
        expected_page = expected["page_number"]

        if expected_page is None:
            if actual_page is not None:
                return False
        else:
            try:
                if int(float(actual_page)) != int(expected_page):
                    return False
            except (TypeError, ValueError):
                return False

    expected_keywords = expected.get("keywords_any", [])
    result_text = normalize(result.get("text"))

    if expected_keywords:
        if not any(
            normalize(keyword) in result_text
            for keyword in expected_keywords
        ):
            return False

    return True


def find_relevant_rank(
    results: list[dict[str, Any]],
    expected: dict[str, Any],
) -> int | None:
    """Return the one-based rank of the first matching passage."""
    for rank, result in enumerate(results, start=1):
        if is_expected_passage(result, expected):
            return rank

    return None


def calculate_metrics(
    ranks: list[int | None],
    total_questions: int,
) -> dict[str, float]:
    """Calculate Recall@5 for one expected passage per query and MRR."""
    hits = sum(
        rank is not None and rank <= K
        for rank in ranks
    )

    reciprocal_ranks = [
        1.0 / rank if rank is not None and rank <= K else 0.0
        for rank in ranks
    ]

    return {
        "recall_at_5": hits / total_questions,
        "mrr_at_5": sum(reciprocal_ranks) / total_questions,
        "hits_at_5": float(hits),
    }


def main() -> None:
    questions = load_questions()

    vector_ranks: list[int | None] = []
    hybrid_ranks: list[int | None] = []
    per_question: list[dict[str, Any]] = []

    print(f"Evaluation questions: {len(questions)}")
    print("Vector candidates:", CANDIDATE_K)
    print("Evaluation cutoff:", K)

    for number, question in enumerate(questions, start=1):
        query = question["query"]
        bid_id = question["bid_id"]
        expected = question["expected"]

        print(
            f"\n[{number}/{len(questions)}] "
            f"{question['id']}: {query}"
        )

        # One semantic search per query. Its top 5 results form the
        # vector-only baseline; the top 10 are also used for RRF.
        semantic_results = semantic_search(
            query=query,
            top_k=CANDIDATE_K,
            bid_id=bid_id,
        )

        vector_results = semantic_results[:K]

        keyword_results = bm25_search(
            query=query,
            top_k=CANDIDATE_K,
            bid_id=bid_id,
        )

        hybrid_results = reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            top_k=K,
        )

        vector_rank = find_relevant_rank(
            vector_results,
            expected,
        )

        hybrid_rank = find_relevant_rank(
            hybrid_results,
            expected,
        )

        vector_ranks.append(vector_rank)
        hybrid_ranks.append(hybrid_rank)

        print("Vector-only relevant rank:", vector_rank or "Not found @5")
        print("Hybrid relevant rank:", hybrid_rank or "Not found @5")

        per_question.append({
            "id": question["id"],
            "query": query,
            "bid_id": bid_id,
            "vector_only_rank": vector_rank,
            "hybrid_rank": hybrid_rank,
            "vector_only_hit_at_5": vector_rank is not None,
            "hybrid_hit_at_5": hybrid_rank is not None,
        })

    total = len(questions)

    vector_metrics = calculate_metrics(vector_ranks, total)
    hybrid_metrics = calculate_metrics(hybrid_ranks, total)

    report = {
        "total_questions": total,
        "cutoff": K,
        "methodology": (
            "One expected source passage per question. "
            "A passage matches when filename, page and at least one "
            "expected keyword match. Recall@5 is hit rate for this "
            "single-target evaluation set."
        ),
        "vector_only": vector_metrics,
        "hybrid": hybrid_metrics,
        "per_question": per_question,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 64)
    print(f"{'Configuration':<18} {'Recall@5':>12} {'MRR@5':>12}")
    print("-" * 44)

    for name, metrics in (
        ("Vector-only", vector_metrics),
        ("Hybrid", hybrid_metrics),
    ):
        print(
            f"{name:<18} "
            f"{metrics['recall_at_5']:>12.3f} "
            f"{metrics['mrr_at_5']:>12.3f}"
        )

    print("\nEvaluation report saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
