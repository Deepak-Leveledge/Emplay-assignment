
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

from groq import Groq
from pydantic import BaseModel, ConfigDict, Field

from emplay_assignment.agents.search_tools import search_bid_documents
from emplay_assignment.config import settings


logger = logging.getLogger(__name__)


class ReconciledChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: str
    original_value: str | None
    revised_value: str | None
    change_type: Literal[
        "added",
        "modified",
        "removed",
        "clarification",
        "deadline_extension",
        "other",
    ]
    change_summary: str
    source_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ReconciliationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: list[ReconciledChange]
    review_notes: list[str]


SEARCH_QUERIES = [
    (
        "addendum_1",
        "Addendum 1 amendments, revisions, corrections, questions and answers, "
        "and changes to the original solicitation or RFP requirements."
    ),
    (
        "addendum_2",
        "Addendum 2 amendments, revised requirements, changed submission deadline, "
        "updated dates, and superseding solicitation instructions."
    ),
    (
        "original_deadline",
        "Original RFP solicitation issue date, original proposal due date, "
        "submission deadline, due time and timezone."
    ),
    (
        "specification_changes",
        "Original and revised product specifications, minimum USB port version, "
        "warranty periods, device requirements, addendum modifications."
    ),
    (
        "other_changes",
        "Changes to bid submission instructions, contract requirements, "
        "mandatory forms, delivery, payment, eligibility or legal conditions."
    ),
]


def is_addendum(item: dict[str, Any]) -> bool:
    """Identify passages coming from a file identified as an addendum."""
    file_name = str(item.get("file_name") or "").lower()
    doc_type = str(item.get("doc_type") or "").lower()

    return (
        "addendum" in file_name
        or "addendum" in doc_type
        or item.get("addendum_number") is not None
    )


def addendum_label(item: dict[str, Any]) -> str | None:
    """Build a label from retrieved source metadata."""
    file_name = str(item.get("file_name") or "")
    match = re.search(r"addendum[\s_-]*(\d+)", file_name, re.IGNORECASE)

    if match:
        return f"Addendum {match.group(1)}"

    number = item.get("addendum_number")
    if number is not None:
        return f"Addendum {number}"

    return None


def retrieve_reconciliation_evidence(
    bid_id: str,
) -> list[dict[str, Any]]:
    """Retrieve relevant original and addendum passages."""
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
                    # Keep the request small enough for the model's token limit.
                    "text": (result.get("text") or "")[:750],
                    "retrieved_for": [],
                }

            retrieved_for = evidence_by_chunk[chunk_id]["retrieved_for"]
            if search_name not in retrieved_for:
                retrieved_for.append(search_name)

    evidence = list(evidence_by_chunk.values())

    for index, item in enumerate(evidence, start=1):
        item["source_id"] = f"S{index}"

    return evidence


def reconcile_addenda(bid_id: str) -> dict[str, Any]:
    """Create a source-grounded change log for the selected bid."""
    if bid_id not in {"Bid1", "Bid2"}:
        raise ValueError("bid_id must be either 'Bid1' or 'Bid2'")

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in .env")

    started = time.perf_counter()
    evidence = retrieve_reconciliation_evidence(bid_id)

    addenda_found = sorted(
        {
            label
            for item in evidence
            if is_addendum(item)
            for label in [addendum_label(item)]
            if label is not None
        }
    )

    def make_result(
        changes: list[dict[str, Any]],
        notes: list[str],
        evidence_count: int,
    ) -> dict[str, Any]:
        return {
            "bid_id": bid_id,
            "agent": "addendum_reconciliation",
            "addenda_found_in_retrieved_evidence": addenda_found,
            "changes": changes,
            "review_notes": notes,
            "evidence_count": evidence_count,
            "latency_ms": round(
                (time.perf_counter() - started) * 1000, 2
            ),
            "model": settings.groq_model,
        }

    if not evidence:
        return make_result(
            [],
            ["No passages were retrieved; reconciliation was not performed."],
            0,
        )

    # Avoid claiming that there are no changes when no addendum evidence
    # could be retrieved in the first place.
    addendum_evidence = [
        item for item in evidence if is_addendum(item)
    ]

    if not addendum_evidence:
        return make_result(
            [],
            [
                "No addendum-labeled passages were retrieved. "
                "This does not prove that the bid has no addenda; "
                "check retrieval coverage before accepting an empty change log."
            ],
            len(evidence),
        )

    evidence_by_id = {
        item["source_id"]: item for item in evidence
    }

    evidence_text = json.dumps(
        evidence,
        ensure_ascii=False,
    )

    system_prompt = """
You are an Addendum Reconciliation Agent for an RFP intelligence platform.

Compare the retrieved original solicitation passages with the retrieved
addendum passages. Identify only changes that are supported by the evidence.

Requirements:
- Every change must cite one or more supplied source IDs.
- Every change must cite at least one passage from an actual addendum file.
- When available, also cite the original passage that establishes the old value.
- Do not invent old values, revised values, requirements, or changes.
- Distinguish a real revision from a clarification or repetition.
- When an addendum supersedes an earlier requirement, report the old value
  and the revised value separately.
- Preserve dates, times, timezones, amounts, model identifiers and units.
- A later addendum takes precedence over an earlier addendum only when the
  documents support that ordering and the later text addresses the same issue.
- Do not assume a change simply because the text differs in unrelated contexts.
- If there is insufficient evidence, omit the proposed change and explain
  the limitation in review_notes.
- If no supported changes are identified, return an empty changes array.
- Do not conclude that the entire bid has no changes merely because only a
  limited number of passages were retrieved.
- Return concise, structured output only.
"""

    user_prompt = f"""
Bid ID: {bid_id}

Retrieved evidence:
{evidence_text}

Identify the documented addendum changes and cite the evidence source IDs.
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
                "name": "addendum_reconciliation",
                "strict": True,
                "schema": ReconciliationDraft.model_json_schema(),
            },
        },
    )

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError(
            "Groq returned an empty reconciliation response"
        )

    draft = ReconciliationDraft.model_validate_json(content)

    validated_changes = []
    review_notes = list(draft.review_notes)

    for change in draft.changes:
        valid_ids = list(
            dict.fromkeys(
                source_id
                for source_id in change.source_ids
                if source_id in evidence_by_id
            )
        )

        cited_addendum_ids = [
            source_id
            for source_id in valid_ids
            if is_addendum(evidence_by_id[source_id])
        ]

        # A purported change without an addendum citation is not accepted.
        if not cited_addendum_ids:
            review_notes.append(
                f"Skipped '{change.field_name}': no valid addendum source cited."
            )
            continue

        sources = []
        for source_id in valid_ids:
            item = evidence_by_id[source_id]
            sources.append(
                {
                    "source_id": source_id,
                    "file_name": item.get("file_name"),
                    "page_number": item.get("page_number"),
                    "doc_type": item.get("doc_type"),
                    "addendum_number": item.get("addendum_number"),
                    "quote": item.get("text", ""),
                }
            )

        validated_changes.append(
            {
                "field_name": change.field_name,
                "original_value": change.original_value,
                "revised_value": change.revised_value,
                "change_type": change.change_type,
                "change_summary": change.change_summary,
                "confidence": change.confidence,
                "source_ids": valid_ids,
                "sources": sources,
            }
        )

    review_notes.append(
        "This change log is based on retrieved passages, not an exhaustive "
        "page-by-page legal review of every source document."
    )

    result = make_result(
        validated_changes,
        review_notes,
        len(evidence),
    )

    logger.info(
        "addendum_reconciliation_completed",
        extra={
            "bid_id": bid_id,
            "changes_count": len(validated_changes),
            "evidence_count": len(evidence),
            "latency_ms": result["latency_ms"],
        },
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile original RFP requirements with addenda."
    )
    parser.add_argument(
        "--bid-id",
        choices=["Bid1", "Bid2"],
        required=True,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file path.",
    )
    args = parser.parse_args()

    result = reconcile_addenda(args.bid_id)

    output_path = Path(
        args.output or f"outputs/{args.bid_id}_addendum_changes.json"
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
