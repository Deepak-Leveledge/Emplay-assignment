
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


# Strict JSON schemas are required by Groq structured outputs.
class ExtractedCommercialLegalField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | None
    source_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None


class CommercialLegalExtractionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bid_bond_requirement: ExtractedCommercialLegalField
    term_of_bid: ExtractedCommercialLegalField
    payment_terms: ExtractedCommercialLegalField
    additional_documentation: ExtractedCommercialLegalField
    contract_or_cooperative: ExtractedCommercialLegalField


# Public field names must match the assignment schema.
FIELD_MAPPING = {
    "Bid Bond Requirement": "bid_bond_requirement",
    "Term of Bid": "term_of_bid",
    "Payment Terms": "payment_terms",
    "Any Additional Documentation Required": "additional_documentation",
    "Contract or Cooperative to use": "contract_or_cooperative",
}


SEARCH_QUERIES = [
    (
        "bid_bond",
        "Bid bond requirement, bid security, bid guarantee, bond amount, "
        "percentage, exemption and submission requirements."
    ),
    (
        "term_of_bid",
        "Contract term, term of bid, contract duration, initial term, "
        "renewal options, extension period and start date."
    ),
    (
        "payment_terms",
        "Payment terms, invoice requirements, payment schedule, "
        "payment deadline, net payment days and invoicing conditions."
    ),
    (
        "additional_documents",
        "Required additional documents, affidavits, forms, certificates, "
        "registration documents, mandatory attachments and bid submission forms."
    ),
    (
        "contract_cooperative",
        "Contract or cooperative purchasing agreement to use, cooperative "
        "contract, purchasing consortium, contract vehicle and cooperative name."
    ),
]


def retrieve_commercial_legal_evidence(bid_id: str) -> list[dict[str, Any]]:
    """Retrieve a small, focused set of evidence passages."""
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
                    # Limit context size to help stay within Groq TPM limits.
                    "text": (result.get("text") or "")[:900],
                    "retrieved_for": [],
                }

            if search_name not in evidence_by_chunk[chunk_id]["retrieved_for"]:
                evidence_by_chunk[chunk_id]["retrieved_for"].append(search_name)

    evidence = list(evidence_by_chunk.values())

    for index, item in enumerate(evidence, start=1):
        item["source_id"] = f"S{index}"

    return evidence


def build_field_value(
    draft: ExtractedCommercialLegalField,
    evidence_by_id: dict[str, dict[str, Any]],
) -> FieldValue:
    """Accept only cited values; otherwise return null with an explanation."""
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
            notes=draft.notes or "Not found explicitly in the retrieved documents.",
        )

    # Do not allow a model-generated value without a real retrieved source.
    if not valid_ids:
        return FieldValue(
            value=None,
            sources=[],
            confidence=0.0,
            notes=(
                "A value was proposed, but no valid supporting citation "
                "was provided. Value omitted."
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


def extract_commercial_legal(bid_id: str) -> dict[str, Any]:
    """Extract the commercial and legal fields for one bid."""
    if bid_id not in {"Bid1", "Bid2"}:
        raise ValueError("bid_id must be either 'Bid1' or 'Bid2'")

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in .env")

    started = time.perf_counter()
    evidence = retrieve_commercial_legal_evidence(bid_id)

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
            "extraction_group": "commercial_legal",
            "fields": fields,
            "evidence_count": 0,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "model": settings.groq_model,
        }

    evidence_by_id = {
        item["source_id"]: item
        for item in evidence
    }

    evidence_text = json.dumps(
        evidence,
        ensure_ascii=False,
    )

    system_prompt = """
You are the Commercial and Legal Extraction Agent for an RFP intelligence platform.

Extract these five fields using ONLY the supplied document evidence:
1. Bid Bond Requirement
2. Term of Bid
3. Payment Terms
4. Any Additional Documentation Required
5. Contract or Cooperative to use

Rules:
- Every non-null value MUST cite one or more supplied source IDs such as S1 or S2.
- Never invent requirements, amounts, dates, durations, payment conditions, or contract names.
- If a field is not explicitly supported by the evidence, return value=null,
  source_ids=[], and explain why in notes.
- Distinguish a bid bond from a performance bond or other security.
- For additional documentation, mention only documents supported by the evidence.
- For contract term, distinguish contract duration from bid submission validity.
  Do not substitute one for the other.
- Preserve meaningful conditions, amounts, percentages and durations in the value.
- If multiple documents describe the same requirement, prefer an explicit
  applicable addendum over superseded original wording.
- Do not claim that a requirement is absent from the entire bid when only a
  limited set of passages was supplied. Use notes such as "Not found in retrieved evidence."
- Confidence must be between 0 and 1 and reflect the evidence quality.
- Return only the requested structured output.
"""

    user_prompt = f"""
Bid ID: {bid_id}

Retrieved evidence, each with a source_id, filename, page and text:
{evidence_text}

Extract the five requested commercial/legal fields.
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
        max_tokens=1400,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "commercial_legal_extraction",
                "strict": True,
                "schema": CommercialLegalExtractionDraft.model_json_schema(),
            },
        },
    )

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty extraction response")

    draft = CommercialLegalExtractionDraft.model_validate_json(content)

    fields = {}

    for field_name, schema_key in FIELD_MAPPING.items():
        fields[field_name] = build_field_value(
            getattr(draft, schema_key),
            evidence_by_id,
        ).model_dump(mode="json")

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    logger.info(
        "commercial_legal_extraction_completed",
        extra={
            "bid_id": bid_id,
            "evidence_count": len(evidence),
            "latency_ms": elapsed_ms,
            "model": settings.groq_model,
        },
    )

    return {
        "bid_id": bid_id,
        "extraction_group": "commercial_legal",
        "fields": fields,
        "evidence_count": len(evidence),
        "latency_ms": elapsed_ms,
        "model": settings.groq_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Commercial & Legal extraction agent."
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

    result = extract_commercial_legal(args.bid_id)

    output_path = Path(
        args.output
        or f"outputs/{args.bid_id}_commercial_legal.json"
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
