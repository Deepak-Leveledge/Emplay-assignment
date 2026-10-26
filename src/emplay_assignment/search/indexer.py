
from __future__ import annotations

import json
import logging
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from pinecone import Pinecone

from emplay_assignment.config import settings
from emplay_assignment.search.embeddings import (
    embed_texts,
    prepare_document_text,
)


logger = logging.getLogger(__name__)

INPUT_FILE = settings.processed_data_dir / "chunks.jsonl"

EMBEDDING_BATCH_SIZE = 8
FETCH_BATCH_SIZE = 100

# Exact dense-vector field name returned by your Pinecone schema.
VECTOR_FIELD = "emplay"


def batches(
    items: list[Any],
    batch_size: int,
) -> Iterator[list[Any]]:
    """Split items into batches."""
    iterator = iter(items)

    while batch := list(islice(iterator, batch_size)):
        yield batch


def load_chunks() -> list[dict[str, Any]]:
    """Load and validate chunks from JSONL."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {INPUT_FILE}"
        )

    chunks: list[dict[str, Any]] = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")

            if not chunk_id:
                raise ValueError(
                    f"Missing chunk_id on line {line_number}"
                )

            if not text.strip():
                logger.warning("Skipping empty chunk: %s", chunk_id)
                continue

            chunks.append(chunk)

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs found.")

    return chunks


def get_pinecone_index():
    """Connect to the existing document-schema index."""
    api_key = settings.pinecone_api_key.get_secret_value().strip()

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    pc = Pinecone(api_key=api_key)

    description = pc.indexes.describe(
        settings.pinecone_index_name
    )

    schema = getattr(description, "schema", None)
    fields = getattr(schema, "fields", None)

    # Handle the SDK's model representation.
    if fields is not None:
        if hasattr(fields, "get"):
            vector_schema = fields.get(VECTOR_FIELD)
        else:
            vector_schema = None
    else:
        vector_schema = None

    if vector_schema is None:
        raise RuntimeError(
            f"Vector field '{VECTOR_FIELD}' was not found. "
            "Inspect the Pinecone index schema before uploading."
        )

    if getattr(vector_schema, "dimension", None) != (
        settings.embedding_dimension
    ):
        raise RuntimeError(
            f"Dimension mismatch for '{VECTOR_FIELD}'."
        )

    status = getattr(description, "status", None)

    if getattr(status, "ready", None) is False:
        raise RuntimeError("Pinecone index is not ready.")

    return pc.index(settings.pinecone_index_name)


def safe_metadata(
    metadata: dict[str, Any],
    chunk_text: str,
) -> dict[str, Any]:
    """Keep scalar metadata and the original passage."""
    allowed_types = (str, int, float, bool)

    result = {
        key: value
        for key, value in metadata.items()
        if value is not None and isinstance(value, allowed_types)
    }

    # The original passage will be used for citations and answers.
    result["chunk_text"] = chunk_text

    return result


def find_existing_ids(
    index,
    chunk_ids: list[str],
    namespace: str,
) -> set[str]:
    """Check existing IDs using the Documents API."""
    existing_ids: set[str] = set()

    for id_batch in batches(chunk_ids, FETCH_BATCH_SIZE):
        response = index.documents.fetch(
            namespace=namespace,
            ids=id_batch,
        )

        existing_ids.update(response.documents.keys())

    return existing_ids


def main() -> None:
    chunks = load_chunks()

    if not chunks:
        raise RuntimeError("No chunks available for indexing.")

    namespace = settings.pinecone_namespace
    index = get_pinecone_index()

    all_ids = [chunk["chunk_id"] for chunk in chunks]

    existing_ids = find_existing_ids(
        index=index,
        chunk_ids=all_ids,
        namespace=namespace,
    )

    pending = [
        chunk
        for chunk in chunks
        if chunk["chunk_id"] not in existing_ids
    ]

    print("Index:", settings.pinecone_index_name)
    print("Namespace:", namespace)
    print("Vector field:", VECTOR_FIELD)
    print("Dimension:", settings.embedding_dimension)
    print("Input chunks:", len(chunks))
    print("Already indexed:", len(existing_ids))
    print("Pending chunks:", len(pending))

    total_upserted = 0

    for batch_number, batch in enumerate(
        batches(pending, EMBEDDING_BATCH_SIZE),
        start=1,
    ):
        embedding_inputs = []

        for chunk in batch:
            metadata = chunk.get("metadata", {})

            filename = metadata.get("file_name", "unknown")
            bid_id = metadata.get("bid_id", "unknown")
            doc_type = metadata.get("doc_type", "other")

            title = (
                f"{bid_id} | {doc_type} | {Path(filename).stem}"
            )

            embedding_inputs.append(
                prepare_document_text(
                    text=chunk["text"],
                    title=title,
                )
            )

        vectors = embed_texts(embedding_inputs)

        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Batch {batch_number}: embedding count mismatch."
            )

        documents = []

        for chunk, vector in zip(batch, vectors):
            if len(vector) != settings.embedding_dimension:
                raise RuntimeError(
                    f"Invalid vector dimension: {chunk['chunk_id']}"
                )

            # Document-schema format:
            # _id = unique ID
            # emplay = 768-dimensional vector
            # remaining top-level fields = metadata
            document = {
                "_id": chunk["chunk_id"],
                VECTOR_FIELD: vector,
                **safe_metadata(
                    chunk.get("metadata", {}),
                    chunk["text"],
                ),
            }

            documents.append(document)

        # IMPORTANT: document-schema index, not index.upsert().
        response = index.documents.upsert(
            namespace=namespace,
            documents=documents,
        )

        uploaded = response.upserted_count
        total_upserted += uploaded

        logger.info(
            "Batch %s: %s documents upserted.",
            batch_number,
            uploaded,
        )

        print(
            f"Batch {batch_number}: {uploaded} documents uploaded; "
            f"total this run: {total_upserted}"
        )

    print("\nUpload requests completed.")
    print("New documents uploaded:", total_upserted)

    # Verify specific IDs using the same Documents API.
    sample_ids = all_ids[:3]

    verification = index.documents.fetch(
        namespace=namespace,
        ids=sample_ids,
        include_fields=["chunk_text", "bid_id", "file_name", "page_number"],
    )

    found_ids = set(verification.documents.keys())

    print("Sample IDs checked:", len(sample_ids))
    print("Sample documents found:", len(found_ids))

    for doc_id, document in verification.documents.items():
        print(
            "Verified:",
            doc_id,
            "| Bid:",
            getattr(document, "bid_id", None),
            "| File:",
            getattr(document, "file_name", None),
            "| Page:",
            getattr(document, "page_number", None),
        )

    if len(found_ids) != len(sample_ids):
        print(
            "WARNING: Some sample records are not visible yet. "
            "Pinecone writes can be eventually consistent; "
            "run the verification again shortly."
        )
    else:
        print("Sample document verification: SUCCESS")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
