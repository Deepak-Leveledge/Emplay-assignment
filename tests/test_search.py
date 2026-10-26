
import json
from pathlib import Path

import emplay_assignment.search.hybrid_search as hybrid_module

from emplay_assignment.search.hybrid_search import (
    bm25_search,
    expand_query,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_preserves_exact_bid_identifier():
    tokens = tokenize("JA-207652")

    assert "ja-207652" in tokens
    assert "ja" in tokens
    assert "207652" in tokens


def test_expand_query_adds_deadline_synonyms():
    expanded = expand_query(
        "What is the submission deadline for Bid1?"
    ).lower()

    assert "submission deadline" in expanded
    assert "due date" in expanded
    assert "closing date" in expanded
    assert "solicitation due" in expanded


def test_expand_query_does_not_remove_original_question():
    query = "What is the warranty for Dell laptops?"
    expanded = expand_query(query)

    assert query in expanded


def test_rrf_ranks_document_found_by_both_searches_higher():
    semantic_results = [
        {
            "chunk_id": "chunk-a",
            "text": "Semantic result A",
            "bid_id": "Bid1",
        },
        {
            "chunk_id": "chunk-b",
            "text": "Semantic result B",
            "bid_id": "Bid1",
        },
    ]

    keyword_results = [
        {
            "chunk_id": "chunk-b",
            "text": "Keyword result B",
            "bid_id": "Bid1",
            "bm25_score": 5.0,
        },
        {
            "chunk_id": "chunk-c",
            "text": "Keyword result C",
            "bid_id": "Bid1",
            "bm25_score": 3.0,
        },
    ]

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=3,
    )

    assert results[0]["chunk_id"] == "chunk-b"
    assert results[0]["semantic_score"] is None
    assert results[0]["bm25_score"] == 5.0
    assert results[0]["rrf_score"] > results[1]["rrf_score"]


def test_bm25_search_finds_exact_bid_number(
    tmp_path: Path,
    monkeypatch,
):
    chunks_file = tmp_path / "chunks.jsonl"

    sample_chunks = [
        {
            "chunk_id": "bid1-rfp",
            "text": (
                "Request For Proposal JA-207652 "
                "Student and Staff Computing Devices."
            ),
            "metadata": {
                "bid_id": "Bid1",
                "file_name": "JA-207652 FINAL.pdf",
                "doc_type": "rfp",
                "page_number": 1,
            },
        },
        {
            "chunk_id": "bid1-addendum",
            "text": (
                "Addendum 2 for RFP JA-207652 "
                "extends the submission due date."
            ),
            "metadata": {
                "bid_id": "Bid1",
                "file_name": "Addendum 2.pdf",
                "doc_type": "addendum",
                "page_number": 1,
            },
        },
        {
            "chunk_id": "bid2-specs",
            "text": (
                "Dell Latitude 5550 with Intel Core Ultra 5 "
                "125U processor and 16 GB RAM."
            ),
            "metadata": {
                "bid_id": "Bid2",
                "file_name": "Dell_Laptop_Specs.pdf",
                "doc_type": "specs",
                "page_number": 1,
            },
        },
    ]

    with chunks_file.open("w", encoding="utf-8") as file:
        for chunk in sample_chunks:
            file.write(json.dumps(chunk) + "\n")

    monkeypatch.setattr(
        hybrid_module,
        "CHUNKS_FILE",
        chunks_file,
    )

    results = bm25_search(
        query="JA-207652",
        top_k=5,
        bid_id="Bid1",
    )

    assert results
    assert results[0]["chunk_id"] == "bid1-rfp"
    assert all(result["bid_id"] == "Bid1" for result in results)


def test_bm25_search_returns_no_results_for_unmatched_query():
    results = bm25_search(
        query="!!!???",
        top_k=5,
    )

    assert results == []
