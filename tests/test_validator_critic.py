
from __future__ import annotations

from emplay_assignment.agents.validator_critic import (
    FIELD_ORDER,
    validate_final_fields,
)
from emplay_assignment.schemas.bid_schema import REQUIRED_BID_FIELDS


def make_valid_fields() -> dict:
    """Build a valid 20-field payload with source citations."""
    citation = {
        "file": "sample_rfp.pdf",
        "page": 1,
        "chunk_id": "chunk-001",
        "quote": "This is a sample supporting passage.",
    }

    return {
        field_name: {
            "value": f"Sample value for {field_name}",
            "sources": [citation.copy()],
            "confidence": 0.9,
            "notes": None,
        }
        for field_name in FIELD_ORDER
    }


def make_addendum_payload() -> dict:
    """Build an empty but structurally valid reconciliation result."""
    return {
        "changes": [],
        "review_notes": [],
        "addenda_found_in_retrieved_evidence": ["Addendum 1"],
    }


def run_validation(
    fields: dict,
    addendum_payload: dict | None = None,
) -> dict:
    return validate_final_fields(
        fields=fields,
        addendum_payload=(
            addendum_payload
            if addendum_payload is not None
            else make_addendum_payload()
        ),
        known_chunk_ids={"chunk-001"},
    )


def test_field_order_matches_assignment_schema():
    assert len(FIELD_ORDER) == 20
    assert len(set(FIELD_ORDER)) == 20
    assert set(FIELD_ORDER) == set(REQUIRED_BID_FIELDS)


def test_valid_20_fields_pass_validation():
    result = run_validation(make_valid_fields())

    assert result["total_fields"] == 20
    assert result["populated_fields_count"] == 20
    assert result["missing_fields_count"] == 0
    assert result["citation_coverage"] == 1.0
    assert result["passed"] is True
    assert result["errors"] == []


def test_missing_required_field_fails_validation():
    fields = make_valid_fields()
    fields.pop("Due Date")

    result = run_validation(fields)

    assert result["passed"] is False
    assert any(
        "Missing required field: Due Date" in error
        for error in result["errors"]
    )


def test_unknown_chunk_id_fails_validation():
    fields = make_valid_fields()
    fields["Bid Number"]["sources"][0]["chunk_id"] = "unknown-chunk"

    result = run_validation(fields)

    assert result["passed"] is False
    assert any(
        "unknown-chunk" in error
        for error in result["errors"]
    )


def test_null_value_with_explanation_is_allowed():
    fields = make_valid_fields()
    fields["Pre Bid Meeting"] = {
        "value": None,
        "sources": [],
        "confidence": 0.0,
        "notes": "Not found in retrieved evidence.",
    }

    result = run_validation(fields)

    assert result["passed"] is True
    assert result["missing_fields_count"] == 1
    assert "Pre Bid Meeting" in result["fields_missing"]


def test_addendum_change_without_addendum_source_is_rejected():
    fields = make_valid_fields()

    addendum_payload = {
        "changes": [
            {
                "field_name": "Due Date",
                "original_value": "June 27",
                "revised_value": "July 9",
                "change_type": "deadline_extension",
                "change_summary": "Deadline updated",
                "confidence": 0.9,
                "source_ids": ["S1"],
                "sources": [
                    {
                        "source_id": "S1",
                        "file_name": "original_rfp.pdf",
                        "page_number": 1,
                        "quote": "Original submission deadline.",
                    }
                ],
            }
        ],
        "review_notes": [],
        "addenda_found_in_retrieved_evidence": ["Addendum 1"],
    }

    result = run_validation(fields, addendum_payload)

    assert result["passed"] is False
    assert any(
        "no cited source" in error.lower()
        for error in result["errors"]
    )
