
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


class ExtractedIdentityField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | None
    source_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None


class IdentitySummaryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bid_number: ExtractedIdentityField
    title: ExtractedIdentityField
    installation: ExtractedIdentityField
    contact_info: ExtractedIdentityField
    company_name: ExtractedIdentityField
    bid_summary: ExtractedIdentityField


FIELD_MAPPING = {
    "Bid Number": "bid_number",
    "Title": "title",
    "Installation": "installation",
    "contact_info": "contact_info",
    "company_name": "company_name",
    "Bid Summary": "bid_summary",
}


SEARCH_QUERIES = [
    (
        "bid_identity",
        "Solicitation number, bid number, RFP number, solicitation title, "
        "project title, bid name and procurement reference number."
    ),
    (
        "installation",
        "Installation requirements, onsite installation, setup, configuration, "
        "deployment, delivery and installation responsibilities."
    ),
    (
        "contact",
        "Procurement contact information, buyer contact, contact person, "
        "procurement email address, phone number and contact details."
    ),
    (
        "organization",
        "Issuing organization, purchasing entity, school district, agency, "
        "buyer name and organization conducting this solicitation."
    ),
    (
        "summary",
        "Purpose of the solicitation, requested products, project scope, "
        "key requirements, eligible products and overview of the bid."
    ),
]


def retrieve_identity_evidence(bid_id: str) -> list[dict[str, Any]]:
    """Retrieve a small, focused set of source passages."""
    evidence_by_chunk: dict[str, dict[str, Any]] = {}

    for search_name, query in SEARCH_QUERIES:
        raw_result = search_bid_documents.invoke(
            {
                "query": query,
                "bid_id": bid_id,
                "top_k": 2,
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
                    "text": (result.get("text") or "")[:800],
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
    draft: ExtractedIdentityField,
    evidence_by_id: dict[str, dict[str, Any]],
) -> FieldValue:
    """Validate that a proposed value references retrieved evidence."""
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
            notes=draft.notes or "Not found in the retrieved evidence.",
        )

    if not valid_ids:
        return FieldValue(
            value=None,
            sources=[],
            confidence=0.0,
            notes=(
                "A value was proposed without a valid supporting citation. "
                "Value omitted."
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
                quote=str(item.get("text") or "")[:800],
            )
        )

    return FieldValue(
        value=draft.value.strip(),
        sources=citations,
        confidence=confidence,
        notes=draft.notes,
    )


def extract_identity_summary(bid_id: str) -> dict[str, Any]:
    """Extract identity, organization, contact, and summary fields."""
    if bid_id not in {"Bid1", "Bid2"}:
        raise ValueError("bid_id must be either 'Bid1' or 'Bid2'")

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in .env")

    started = time.perf_counter()
    evidence = retrieve_identity_evidence(bid_id)

    if not evidence:
        fields = {
            field_name: FieldValue(
                value=None,
                sources=[],
                confidence=0.0,
                notes="No supporting passages were retrieved.",
            ).model_dump(mode="json")
            for field_name in FIELD_MAPPING
        }

        return {
            "bid_id": bid_id,
            "extraction_group": "identity_summary",
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
You are the Identity and Summary Extraction Agent for an RFP intelligence platform.

Extract these six fields using only the supplied evidence:
1. Bid Number
2. Title
3. Installation
4. contact_info
5. company_name
6. Bid Summary

Rules:
- Every non-null value must cite one or more supplied source IDs such as S1.
- Never invent a bid number, title, organization, person's name, email or phone.
- Preserve official names and solicitation identifiers accurately.
- company_name means the issuing/purchasing organization when the evidence
  identifies it; do not treat a bidder or product manufacturer as the buyer.
- contact_info should contain only contact details explicitly supported by
  the passages.
- Installation should describe actual installation requirements or state
  that the requirement was not found in the retrieved passages.
- Bid Summary should be concise: approximately 2 to 4 sentences describing
  the documented purpose and scope of the solicitation.
- The summary must not introduce facts absent from the evidence.
- If a field cannot be supported, return value=null, source_ids=[],
  and explain the limitation in notes.
- Do not claim a value is absent from every source document merely because it
  was not present in the retrieved passages.
- Confidence must be between 0 and 1.
- Return only the requested structured output.
"""

    user_prompt = f"""
Bid ID: {bid_id}

Retrieved evidence with source IDs, filenames and page numbers:
{evidence_text}

Extract the six requested fields.
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
        max_tokens=2200,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "identity_summary_extraction",
                "strict": True,
                "schema": IdentitySummaryDraft.model_json_schema(),
            },
        },
    )

    content = completion.choices[0].message.content

    if not content:
        raise RuntimeError("Groq returned an empty extraction response")

    draft = IdentitySummaryDraft.model_validate_json(content)

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
        "identity_summary_extraction_completed",
        extra={
            "bid_id": bid_id,
            "evidence_count": len(evidence),
            "latency_ms": elapsed_ms,
            "model": settings.groq_model,
        },
    )

    return {
        "bid_id": bid_id,
        "extraction_group": "identity_summary",
        "fields": fields,
        "evidence_count": len(evidence),
        "latency_ms": elapsed_ms,
        "model": settings.groq_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the identity and summary extraction agent."
    )
    parser.add_argument(
        "--bid-id",
        choices=["Bid1", "Bid2"],
        required=True,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path.",
    )
    args = parser.parse_args()

    result = extract_identity_summary(args.bid_id)

    output_path = Path(
        args.output or f"outputs/{args.bid_id}_identity_summary.json"
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
