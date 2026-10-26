
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from emplay_assignment.config import settings
from emplay_assignment.search.semantic_search import semantic_search


CHUNKS_FILE = settings.processed_data_dir / "chunks.jsonl"
RRF_K = 60

# Preserve identifiers such as JA-207652 and split their components too.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "when", "where", "which", "who", "how",
    "for", "in", "on", "of", "to", "by", "at", "from",
    "with", "please", "tell", "me", "does", "do",
}

QUERY_EXPANSIONS = [
    (
        re.compile(
            r"\b(deadline|due date|closing date|submission deadline"
            r"|solicitation due)\b",
            re.IGNORECASE,
        ),
        "submission deadline due date closing date "
        "solicitation due final due date proposal deadline",
    ),
    (
        re.compile(r"\bbid bond|bid security\b", re.IGNORECASE),
        "bid bond bid security proposal bond surety bond",
    ),
    (
        re.compile(r"\baffidavit|affidavits\b", re.IGNORECASE),
        "affidavit affidavits sworn statement certification forms",
    ),
    (
        re.compile(r"\bwarranty|warranties\b", re.IGNORECASE),
        "warranty warranties extended warranty coverage manufacturer warranty",
    ),
    (
        re.compile(r"\bmodel number|model_no\b", re.IGNORECASE),
        "model number model no product model",
    ),
    (
        re.compile(r"\bpart number|part_no|\bsku\b", re.IGNORECASE),
        "part number part no SKU product identifier",
    ),
    (
        re.compile(r"\bdelivery|delivery date\b", re.IGNORECASE),
        "delivery date delivery window shipment shipping",
    ),
    (
        re.compile(r"\bpayment terms|payment\b", re.IGNORECASE),
        "payment terms invoicing invoice net 30",
    ),
]


def expand_query(query: str) -> str:
    """Add related procurement terms while preserving the original query."""
    expansions = []

    for pattern, terms in QUERY_EXPANSIONS:
        if pattern.search(query):
            expansions.append(terms)

    return " ".join([query, *expansions])


def tokenize(text: str) -> list[str]:
    """Normalize text and retain exact identifiers such as JA-207652."""
    text = text.lower()

    for dash in ("–", "—", "−"):
        text = text.replace(dash, "-")

    tokens = TOKEN_PATTERN.findall(text)
    output = []

    for token in tokens:
        if token not in STOPWORDS:
            output.append(token)

        parts = re.split(r"[-_./]", token)

        for part in parts:
            if part and part not in STOPWORDS and part != token:
                output.append(part)

    return output



def load_chunks(bid_id: str | None = None) -> list[dict[str, Any]]:
    """Load indexed-source chunks for the BM25 keyword search."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_FILE}"
        )

    chunks: list[dict[str, Any]] = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            if not chunk.get("text", "").strip():
                continue

            metadata = chunk.get("metadata", {})

            if bid_id and metadata.get("bid_id") != bid_id:
                continue

            chunk["metadata"] = metadata
            chunks.append(chunk)

    return chunks


def bm25_search(
    query: str,
    top_k: int = 10,
    bid_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve keyword-matching chunks using BM25."""
    expanded_query = expand_query(query)
    query_tokens = tokenize(expanded_query)

    if not query_tokens:
        return []

    chunks = load_chunks(bid_id=bid_id)

    if not chunks:
        return []

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    query_token_set = set(query_tokens)

    candidates: list[dict[str, Any]] = []

    for chunk, tokens, score in zip(chunks, tokenized_corpus, scores):
        # Exclude documents without any actual query-token overlap.
        overlap = len(query_token_set.intersection(tokens))

        if overlap == 0:
            continue

        metadata = chunk["metadata"]

        candidates.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "bid_id": metadata.get("bid_id"),
            "file_name": metadata.get("file_name"),
            "doc_type": metadata.get("doc_type"),
            "page_number": metadata.get("page_number"),
            "addendum_number": metadata.get("addendum_number"),
            "document_date": metadata.get("document_date"),
            "bm25_score": float(score),
            "lexical_overlap": overlap,
        })

    # BM25 score is the primary ranking signal.
    # Lexical overlap breaks ties in favour of direct term matches.
    candidates.sort(
        key=lambda result: (
            result["bm25_score"],
            result["lexical_overlap"],
        ),
        reverse=True,
    )

    return candidates[:top_k]


def reciprocal_rank_fusion(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Merge semantic and BM25 rankings using Reciprocal Rank Fusion."""
    combined: dict[str, dict[str, Any]] = {}

    result_sets = [
        ("semantic", semantic_results),
        ("keyword", keyword_results),
    ]

    for source, results in result_sets:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in combined:
                combined[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get("text", ""),
                    "bid_id": result.get("bid_id"),
                    "file_name": result.get("file_name"),
                    "doc_type": result.get("doc_type"),
                    "page_number": result.get("page_number"),
                    "addendum_number": result.get("addendum_number"),
                    "document_date": result.get("document_date"),
                    "semantic_score": None,
                    "bm25_score": None,
                    "lexical_overlap": 0,
                    "rrf_score": 0.0,
                }

            entry = combined[chunk_id]

            # RRF combines ranks, not raw semantic/BM25 scores.
            entry["rrf_score"] += 1.0 / (RRF_K + rank)

            if source == "semantic":
                entry["semantic_score"] = result.get("score")

                # Preserve text from either retrieval source.
                if result.get("text"):
                    entry["text"] = result["text"]

            else:
                entry["bm25_score"] = result.get("bm25_score")
                entry["lexical_overlap"] = result.get(
                    "lexical_overlap", 0
                )

                if not entry["text"]:
                    entry["text"] = result.get("text", "")

    ranked = sorted(
        combined.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )

    return ranked[:top_k]


def hybrid_search(
    query: str,
    top_k: int = 5,
    bid_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run semantic + keyword search, then merge their rankings."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    candidate_k = min(max(top_k * 2, 10), 20)

    semantic_results = semantic_search(
        query=query,
        top_k=candidate_k,
        bid_id=bid_id,
    )

    keyword_results = bm25_search(
        query=query,
        top_k=candidate_k,
        bid_id=bid_id,
    )

    return reciprocal_rank_fusion(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=top_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid semantic and BM25 search over RFP documents."
    )

    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--bid-id",
        choices=["Bid1", "Bid2"],
        default=None,
    )
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    results = hybrid_search(
        query=args.query,
        top_k=args.top_k,
        bid_id=args.bid_id,
    )

    if not results:
        print("No matching passages found.")
        return

    print(f"\nQuery: {args.query}")
    print(f"Results: {len(results)}")

    for rank, result in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"Rank: {rank}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(f"Semantic score: {result['semantic_score']}")
        print(f"BM25 score: {result['bm25_score']}")
        print(f"Lexical overlap: {result['lexical_overlap']}")
        print(f"Bid: {result['bid_id']}")
        print(f"File: {result['file_name']}")
        print(f"Document type: {result['doc_type']}")
        print(f"PDF page: {result['page_number']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print("\nPassage:")
        print(result["text"] or "[No passage text found]")


if __name__ == "__main__":
    main()
