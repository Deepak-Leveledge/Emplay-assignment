
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from emplay_assignment.config import settings


logger = logging.getLogger(__name__)

CHUNK_SIZE = 2800
CHUNK_OVERLAP = 350

INPUT_FILE = settings.processed_data_dir / "normalized_documents.jsonl"
OUTPUT_FILE = settings.processed_data_dir / "chunks.jsonl"


def find_chunk_boundary(
    text: str,
    start: int,
    proposed_end: int,
) -> int:
    """Prefer a line boundary without producing very small chunks."""
    if proposed_end >= len(text):
        return len(text)

    minimum_boundary = start + int((proposed_end - start) * 0.65)

    newline_boundary = text.rfind(
        "\n",
        minimum_boundary,
        proposed_end,
    )

    if newline_boundary > start:
        return newline_boundary

    space_boundary = text.rfind(
        " ",
        minimum_boundary,
        proposed_end,
    )

    if space_boundary > start:
        return space_boundary

    return proposed_end


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split text into overlapping chunks using character boundaries."""
    text = text.strip()

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    if len(text) <= chunk_size:
        return [{"text": text, "char_start": 0, "char_end": len(text)}]

    chunks: list[dict[str, Any]] = []
    start = 0

    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = find_chunk_boundary(text, start, proposed_end)

        # Prevent an invalid or extremely short progress step.
        if end <= start:
            end = proposed_end

        piece = text[start:end].strip()

        if piece:
            chunks.append({
                "text": piece,
                "char_start": start,
                "char_end": end,
            })

        if end >= len(text):
            break

        # Begin the next chunk before the previous one ends.
        next_start = max(start + 1, end - overlap)

        # Prefer the start of a following line, when possible.
        newline = text.find("\n", next_start, end)

        if newline != -1:
            next_start = newline + 1

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def make_chunk_id(
    document_id: str,
    chunk_index: int,
    text: str,
) -> str:
    """Generate a reproducible chunk ID."""
    source = f"{document_id}|{chunk_index}|{text}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


def chunk_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunk a normalized document while retaining source metadata."""
    text = record.get("text", "")
    metadata = record.get("metadata", {})
    document_id = record.get("doc_id", "")

    if not text.strip():
        logger.warning(
            "Skipping empty source record: %s",
            document_id,
        )
        return []

    text_parts = split_text(text)

    output: list[dict[str, Any]] = []

    for index, part in enumerate(text_parts):
        chunk_text = part["text"]

        output.append({
            "chunk_id": make_chunk_id(
                document_id,
                index,
                chunk_text,
            ),
            "doc_id": document_id,
            "chunk_index": index,
            "text": chunk_text,
            "char_start": part["char_start"],
            "char_end": part["char_end"],
            "metadata": {
                **metadata,
                "chunk_index": index,
                "source_doc_id": document_id,
                "chunking_strategy": "character_window_with_overlap",
            },
        })

    return output


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Normalized documents not found: {INPUT_FILE}. "
            "Run the parser first."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    total_chunks = 0
    skipped_records = 0
    lengths: list[int] = []
    chunk_ids: set[str] = set()

    with INPUT_FILE.open("r", encoding="utf-8") as source, \
         OUTPUT_FILE.open("w", encoding="utf-8") as destination:

        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} of {INPUT_FILE}"
                ) from exc

            total_records += 1
            chunks = chunk_record(record)

            if not chunks:
                skipped_records += 1

            for chunk in chunks:
                if chunk["chunk_id"] in chunk_ids:
                    raise ValueError(
                        f"Duplicate chunk ID detected: {chunk['chunk_id']}"
                    )

                chunk_ids.add(chunk["chunk_id"])

                destination.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )

                total_chunks += 1
                lengths.append(len(chunk["text"]))

    average_length = (
        round(sum(lengths) / len(lengths), 1)
        if lengths
        else 0
    )

    logger.info("Source records: %s", total_records)
    logger.info("Chunks created: %s", total_chunks)
    logger.info("Empty records skipped: %s", skipped_records)
    logger.info("Average chunk characters: %s", average_length)
    logger.info("Maximum chunk characters: %s", max(lengths, default=0))
    logger.info("Output file: %s", OUTPUT_FILE)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
