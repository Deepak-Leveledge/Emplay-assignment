
from __future__ import annotations

import argparse

from pinecone import DenseVectorQuery, Pinecone

from emplay_assignment.config import settings
from emplay_assignment.search.embeddings import (
    embed_texts,
    prepare_query_text,
)


VECTOR_FIELD = "emplay"

INCLUDE_FIELDS = [
    "chunk_text",
    "bid_id",
    "file_name",
    "doc_type",
    "page_number",
    "addendum_number",
    "document_date",
]


def create_index():
    """Connect to the existing Pinecone index."""
    api_key = settings.pinecone_api_key.get_secret_value().strip()

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from .env")

    pc = Pinecone(api_key=api_key)
    return pc.index(settings.pinecone_index_name)


def semantic_search(
    query: str,
    top_k: int = 5,
    bid_id: str | None = None,
) -> list[dict]:
    """Retrieve relevant passages using Gemini embeddings + Pinecone."""
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    # Step 1: Convert the user's question into a query embedding.
    query_text = prepare_query_text(query)
    query_vector = embed_texts([query_text])[0]

    if len(query_vector) != settings.embedding_dimension:
        raise ValueError(
            f"Query embedding dimension mismatch: "
            f"expected {settings.embedding_dimension}, "
            f"received {len(query_vector)}"
        )

    # Step 2: Connect to Pinecone.
    index = create_index()

    # Step 3: Search using the same dense-vector field used during indexing.
    search_args = {
        "namespace": settings.pinecone_namespace,
        "top_k": top_k,
        "score_by": [
            DenseVectorQuery(
                field=VECTOR_FIELD,
                values=query_vector,
            )
        ],
        "include_fields": INCLUDE_FIELDS,
    }

    # Optional filter: restrict results to one bid.
    if bid_id:
        search_args["filter"] = {
            "bid_id": {"$eq": bid_id}
        }

    response = index.documents.search(**search_args)

    # Step 4: Convert matches into a clean, citation-friendly structure.
    results = []

    for match in response.matches:
        results.append({
            "chunk_id": match.id,
            "score": match.score,
            "text": match.get("chunk_text", ""),
            "bid_id": match.get("bid_id"),
            "file_name": match.get("file_name"),
            "doc_type": match.get("doc_type"),
            "page_number": match.get("page_number"),
            "addendum_number": match.get("addendum_number"),
            "document_date": match.get("document_date"),
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search indexed RFP documents semantically."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question or search query.",
    )

    parser.add_argument(
        "--bid-id",
        choices=["Bid1", "Bid2"],
        default=None,
        help="Optional bid filter.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve (1-20).",
    )

    args = parser.parse_args()

    results = semantic_search(
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
        print(f"Similarity score: {result['score']}")
        print(f"Bid: {result['bid_id']}")
        print(f"File: {result['file_name']}")
        print(f"Document type: {result['doc_type']}")
        print(f"PDF page: {result['page_number']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print("\nPassage:")
        print(result["text"] or "[No stored text found]")


if __name__ == "__main__":
    main()
