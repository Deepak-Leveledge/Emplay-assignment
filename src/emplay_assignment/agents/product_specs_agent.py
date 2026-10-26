
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from groq import Groq
from pydantic import BaseModel, ConfigDict, Field

from emplay_assignment.agents.search_tools import search_bid_documents
from emplay_assignment.config import settings
from emplay_assignment.schemas.bid_schema import FieldValue, SourceCitation


logger = logging.getLogger(__name__)


class ExtractedProductField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | None
    source_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None


class ProductSpecsExtractionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manufacturer_for_registration: ExtractedProductField
    model_no: ExtractedProductField
    part_no: ExtractedProductField
    product: ExtractedProductField
    product_specification: ExtractedProductField


FIELD_MAPPING = {
    "MFG for Registration": "manufacturer_for_registration",
    "Model_no": "model_no",
    "Part_no": "part_no",
    "Product": "product",
    "Product Specification": "product_specification",
}


SEARCH_QUERIES = [
    (
        "manufacturer_registration",
        "Manufacturer or MFG registration requirements, manufacturer "
        "authorization, OEM registration, manufacturer name and "
        "manufacturer eligibility requirements."
    ),
    (
        "model_number",
        "Exact product model number, Model No, model identifier, "
        "laptop model, device model and model-specific requirements."
    ),
    (
        "part_number",
        "Exact part number, Part No, manufacturer part number, SKU, "
        "product identifier and ordering part number."
    ),
    (
        "product",
        "Products requested in this bid, product categories, equipment "
        "types, laptop or computing device types, quantities and intended use."
    ),
    (
        "product_specification",
        "Technical product specifications, processor, memory, storage, "
        "display, ports, operating system, warranty and minimum hardware "
        "requirements, including changes in addenda."
    ),
]


def retrieve_product_evidence(bid_id: str) -> list[dict[str, Any]]:
    """Retrieve focused evidence passages from the selected bid."""
    evidence_by_chunk: dict[str, dict[str, Any]] = {}

    for search_name, query in SEARCH_QUERIES:
        raw_result = search_bid_documents.invoke(
            {
                "query": query,
                "bid_id": bid_id,
                "top_k": 3,
            }
        )

        payload = (
            json.loads(raw_result)
            if isinstance(raw_result, str)
            else raw_result
        )

        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected search response for {search_name}"
            )

        if payload.get("error"):
            raise RuntimeError(
                f"Search failed for {search_name}: {payload['error']}"
            )

        for result in payload.get("results", []):
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue

            if chunk_id not in evidence_by_chunk:
                evidence_by_chunk[chunk_id] = {
                    "chunk_id": chunk_id,
                    "bid_id": result.get("bid_id", bid_id),
                    "file_name": result.get("file_name"),
                    "doc_type": result.get("doc_type"),
                    "page_number": result.get("page_number"),
                    "addendum_number": result.get("addendum_number"),
                    "text": (result.get("text") or "")[:900],
                    "retrieved_for": [],
                }

            retrieved_for = evidence_by_chunk[chunk_id]["retrieved_for"]
            if search_name not in retrieved_for:
                retrieved_for.append(search_name)

    evidence = list(evidence_by_chunk.values())

    for index, item in enumerate(evidence, start=1):
        item["source_id"] = f"S{index}"

    return evidence


def build_field_value(
    draft: ExtractedProductField,
    evidence_by_id: dict[str, dict[str, Any]],
) -> FieldValue:
    """Require a valid retrieved source for every non-null value."""
    confidence = max(0.0, min(1.0, draft.confidence or 0.0))

    valid_ids = list(
        dict.fromkeys(
            source_id
            for source_id in draft.source_ids
            if source_id in evidence_by_id
        )
    )

    if draft.value is None or not draft.value.strip():
        return FieldValue(
            value=None,
            sources=[],
            confidence=confidence,
            notes=draft.notes
            or "Not found explicitly in the retrieved evidence.",
        )

    if not valid_ids:
        return FieldValue(
            value=None,
            sources=[],
            confidence=0.0,
            notes=(
                "A value was proposed, but no valid supporting source "
                "was supplied. Value omitted."
            ),
        )

    citations = []

    for source_id in valid_ids:
        item = evidence_by_id[source_id]

        citations.append(
            SourceCitation(
                file=str(item.get("file_name") or "Unknown source"),
                page=item.get("page_number"),
                chunk_id=str(item["chunk_id"]),
                quote=str(item.get("text") or "")[:900],
            )
        )

    return FieldValue(
        value=draft.value.strip(),
        sources=citations,
        confidence=confidence,
        notes=draft.notes,
    )


def extract_product_specs(bid_id: str) -> dict[str, Any]:
    """Extract product and specification fields for one bid."""
    if bid_id not in {"Bid1", "Bid2"}:
        raise ValueError("bid_id must be either 'Bid1' or 'Bid2'")

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in .env")

    started = time.perf_counter()
    evidence = retrieve_product_evidence(bid_id)

    if not evidence:
        fields = {
            field_name: FieldValue(
                value=None,
                sources=[],
                confidence=0.0,
                notes="No supporting passages were retrieved from the bid documents.",
            ).model_dump(mode="json")
            for field_name in FIELD_MAPPING
        }

        return {
            "bid_id": bid_id,
            "extraction_group": "product_specifications",
            "fields": fields,
            "evidence_count": 0,
            "latency_ms": round(
                (time.perf_counter() - started) * 1000, 2
            ),
            "model": settings.groq_model,
        }

    evidence_by_id = {
        item["source_id"]: item for item in evidence
    }

    evidence_text = json.dumps(
        evidence,
        ensure_ascii=False,
    )

    system_prompt = """
You are the Product and Specifications Extraction Agent for an RFP platform.

Extract the following fields using only the supplied evidence:
1. MFG for Registration
2. Model_no
3. Part_no
4. Product
5. Product Specification

Rules:
- Every non-null value must cite one or more supplied source IDs, such as S1.
- Never invent a manufacturer, model number, part number, product, or specification.
- Preserve exact model and part numbers, including letters, digits, hyphens,
  and suffixes. Do not silently normalize their spelling.
- Distinguish an explicitly identified product model from a minimum technical
  requirement. Do not treat a required specification as a confirmed model number.
- For product specifications, preserve important values and units, including
  processor, RAM, storage, screen size, ports, operating system and warranty
  when supported by the passages.
- If the bid requests multiple product types or configurations, describe them
  clearly rather than arbitrarily selecting one.
- For MFG for Registration, use the meaning supported by the documents.
  Do not assume this means a specific manufacturer if the source does not say so.
- Prefer an applicable addendum when it changes an original specification.
- If the supplied evidence does not support a field, return value=null,
  source_ids=[], and explain the limitation in notes.
- Do not claim a field is absent from all bid documents merely because it is
  absent from the retrieved passages.
- Confidence must be between 0 and 1 and reflect evidence quality.
- Return only the requested structured output.
"""

    user_prompt = f"""
Bid ID: {bid_id}

Retrieved source passages:
{evidence_text}

Extract all five requested product fields.
"""

    client = Groq(
        api_key=settings.groq_api_key.get_secret_value()
    )

    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=2400,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "product_specs_extraction",
                "strict": True,
                "schema": ProductSpecsExtractionDraft.model_json_schema(),
            },
        },
    )

    content = completion.choices[0].message.content

    if not content:
        raise RuntimeError("Groq returned an empty extraction response")

    draft = ProductSpecsExtractionDraft.model_validate_json(content)

    fields = {}

    for field_name, schema_key in FIELD_MAPPING.items():
        fields[field_name] = build_field_value(
            getattr(draft, schema_key),
            evidence_by_id,
        ).model_dump(mode="json")

    elapsed_ms = round(
        (time.perf_counter() - started) * 1000, 2
    )

    logger.info(
        "product_specs_extraction_completed",
        extra={
            "bid_id": bid_id,
            "evidence_count": len(evidence),
            "latency_ms": elapsed_ms,
            "model": settings.groq_model,
        },
    )

    return {
        "bid_id": bid_id,
        "extraction_group": "product_specifications",
        "fields": fields,
        "evidence_count": len(evidence),
        "latency_ms": elapsed_ms,
        "model": settings.groq_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Product & Specifications extraction agent."
    )
    parser.add_argument(
        "--bid-id",
        choices=["Bid1", "Bid2"],
        required=True,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the output JSON file.",
    )
    args = parser.parse_args()

    result = extract_product_specs(args.bid_id)

    output_path = Path(
        args.output or f"outputs/{args.bid_id}_product_specs.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved output: {output_path}")


if __name__ == "__main__":
    main()
