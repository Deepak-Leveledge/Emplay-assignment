
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from emplay_assignment.config import PROJECT_ROOT
from emplay_assignment.schemas.bid_schema import (
    REQUIRED_BID_FIELDS,
    FieldValue,
)


FIELD_ORDER = [
    "Bid Number",
    "Title",
    "Due Date",
    "Bid Submission Type",
    "Term of Bid",
    "Pre Bid Meeting",
    "Installation",
    "Bid Bond Requirement",
    "Delivery Date",
    "Payment Terms",
    "Any Additional Documentation Required",
    "MFG for Registration",
    "Contract or Cooperative to use",
    "Model_no",
    "Part_no",
    "Product",
    "contact_info",
    "company_name",
    "Bid Summary",
    "Product Specification",
]


def load_known_chunk_ids() -> tuple[set[str] | None, str | None]:
    """
    Load chunk IDs from the local processed corpus.

    None means the chunk file could not be checked.
    An empty set means the file was read but contained no chunk IDs.
    """
    chunks_file = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

    if not chunks_file.exists():
        return None, f"Chunk file not found: {chunks_file}"

    known_ids: set[str] = set()

    try:
        with chunks_file.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk_id = (
                    record.get("chunk_id")
                    or record.get("id")
                    or record.get("_id")
                )

                if chunk_id:
                    known_ids.add(str(chunk_id))

    except OSError as exc:
        return None, f"Could not read chunk file: {exc}"

    if not known_ids:
        return None, "Chunk file was readable, but no chunk IDs were found."

    return known_ids, None


def validate_addendum_log(
    addendum_payload: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Validate the structure and source attribution of addendum changes."""
    errors: list[str] = []
    warnings: list[str] = []

    changes = addendum_payload.get("changes", [])

    if not isinstance(changes, list):
        return ["Addendum change log must contain a list named 'changes'."], warnings

    for index, change in enumerate(changes):
        label = f"Addendum change #{index + 1}"

        if not isinstance(change, dict):
            errors.append(f"{label} is not an object.")
            continue

        source_ids = change.get("source_ids") or []
        sources = change.get("sources") or []

        if not source_ids:
            errors.append(f"{label} has no source_ids.")
        if not sources:
            errors.append(f"{label} has no source records.")

        if not isinstance(sources, list):
            errors.append(f"{label} sources must be a list.")
            continue

        source_id_set = set(str(item) for item in source_ids)
        found_addendum_file = False

        for source in sources:
            if not isinstance(source, dict):
                errors.append(f"{label} contains a malformed source.")
                continue

            source_id = str(source.get("source_id") or "")
            file_name = str(source.get("file_name") or "")
            quote = str(source.get("quote") or "")

            if not source_id:
                errors.append(f"{label} contains a source without source_id.")
            elif source_id_set and source_id not in source_id_set:
                errors.append(
                    f"{label} has a source record whose ID is absent from source_ids."
                )

            if not file_name:
                errors.append(f"{label} contains a source without file_name.")

            if not quote.strip():
                warnings.append(
                    f"{label} contains a source with an empty quote."
                )

            if "addendum" in file_name.lower():
                found_addendum_file = True

        if sources and not found_addendum_file:
            errors.append(
                f"{label} has no cited source whose filename identifies it as an addendum."
            )

    if not changes and not addendum_payload.get(
        "addenda_found_in_retrieved_evidence"
    ):
        warnings.append(
            "No addendum-labeled passages were retrieved. An empty change log "
            "does not prove that the bid has no addenda."
        )

    for note in addendum_payload.get("review_notes", []) or []:
        warnings.append(f"Reconciliation note: {note}")

    return errors, warnings


def validate_final_fields(
    fields: dict[str, Any],
    addendum_payload: dict[str, Any],
    known_chunk_ids: set[str] | None,
    chunk_check_note: str | None = None,
    inherited_errors: list[str] | None = None,
) -> dict[str, Any]:
    """
    Critic checks:
    - exact field coverage
    - each field conforms to FieldValue
    - non-null values have valid source citations
    - citation chunk IDs exist in the processed corpus when available
    - addendum changes have source attribution
    """
    errors = list(inherited_errors or [])
    warnings: list[str] = []
    invalid_fields: list[str] = []
    populated_fields: list[str] = []
    missing_fields: list[str] = []
    cited_populated_fields: list[str] = []

    expected = set(FIELD_ORDER)
    actual = set(fields)

    for field_name in FIELD_ORDER:
        if field_name not in fields:
            errors.append(f"Missing required field: {field_name}")
            invalid_fields.append(field_name)

    for field_name in sorted(actual - expected):
        warnings.append(f"Unexpected field is present: {field_name}")

    if set(FIELD_ORDER) != set(REQUIRED_BID_FIELDS):
        errors.append(
            "FIELD_ORDER does not match REQUIRED_BID_FIELDS in bid_schema.py."
        )

    for field_name in FIELD_ORDER:
        if field_name not in fields:
            continue

        try:
            field_value = FieldValue.model_validate(fields[field_name])
        except ValidationError as exc:
            errors.append(
                f"{field_name} failed FieldValue validation: {exc.errors()}"
            )
            invalid_fields.append(field_name)
            continue

        if field_value.value is None:
            missing_fields.append(field_name)
            if not (field_value.notes or "").strip():
                errors.append(
                    f"{field_name} is null but has no explanation in notes."
                )
                invalid_fields.append(field_name)
            continue

        populated_fields.append(field_name)
        field_has_invalid_citation = False

        if not field_value.sources:
            errors.append(
                f"{field_name} has a value but no source citations."
            )
            field_has_invalid_citation = True

        for citation in field_value.sources:
            if not citation.file.strip():
                errors.append(f"{field_name} has a citation without a filename.")
                field_has_invalid_citation = True

            if not citation.chunk_id.strip():
                errors.append(f"{field_name} has a citation without a chunk_id.")
                field_has_invalid_citation = True

            if not citation.quote.strip():
                errors.append(f"{field_name} has an empty citation quote.")
                field_has_invalid_citation = True

            if (
                known_chunk_ids is not None
                and citation.chunk_id not in known_chunk_ids
            ):
                errors.append(
                    f"{field_name} cites unknown chunk_id: {citation.chunk_id}"
                )
                field_has_invalid_citation = True

        if not field_has_invalid_citation and field_value.sources:
            cited_populated_fields.append(field_name)

        if field_has_invalid_citation:
            invalid_fields.append(field_name)

    addendum_errors, addendum_warnings = validate_addendum_log(
        addendum_payload
    )
    errors.extend(addendum_errors)
    warnings.extend(addendum_warnings)

    if chunk_check_note:
        warnings.append(chunk_check_note)

    total_fields = len(FIELD_ORDER)
    populated_count = len(populated_fields)
    missing_count = len(missing_fields)

    citation_coverage = (
        round(len(cited_populated_fields) / populated_count, 3)
        if populated_count
        else 0.0
    )

    # Missing values can be legitimate when documents do not state a field.
    # Structural errors and invalid citations fail validation; missing fields
    # are reported separately for human review.
    return {
        "total_fields": total_fields,
        "populated_fields_count": populated_count,
        "missing_fields_count": missing_count,
        "fields_missing": missing_fields,
        "invalid_fields": sorted(set(invalid_fields)),
        "fields_with_valid_citations": len(cited_populated_fields),
        "citation_coverage": citation_coverage,
        "addendum_changes_count": len(
            addendum_payload.get("changes", []) or []
        ),
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "retry_candidates": sorted(set(invalid_fields)),
    }
