
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emplay_assignment.config import PROJECT_ROOT
from emplay_assignment.agents.validator_critic import (
    FIELD_ORDER,
    load_known_chunk_ids,
)


def check_bid(
    bid_id: str,
    known_chunk_ids: set[str] | None,
) -> dict[str, Any]:
    output_path = PROJECT_ROOT / "outputs" / f"{bid_id}_final.json"

    errors: list[str] = []
    warnings: list[str] = []
    populated_count = 0
    null_count = 0
    valid_citation_count = 0
    citation_count = 0

    if not output_path.exists():
        return {
            "bid_id": bid_id,
            "file": str(output_path),
            "passed": False,
            "errors": [f"Final JSON file not found: {output_path}"],
            "warnings": [],
        }

    try:
        with output_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "bid_id": bid_id,
            "file": str(output_path),
            "passed": False,
            "errors": [f"Could not read valid JSON: {exc}"],
            "warnings": [],
        }

    if not isinstance(payload, dict):
        return {
            "bid_id": bid_id,
            "file": str(output_path),
            "passed": False,
            "errors": ["Final output must be a JSON object."],
            "warnings": [],
        }

    if payload.get("bid_id") != bid_id:
        errors.append(
            f"Expected bid_id {bid_id}, got {payload.get('bid_id')!r}."
        )

    fields = payload.get("fields")

    if not isinstance(fields, dict):
        errors.append("'fields' must be a JSON object.")
        fields = {}

    expected = set(FIELD_ORDER)
    actual = set(fields)

    missing_keys = sorted(expected - actual)
    extra_keys = sorted(actual - expected)

    if missing_keys:
        errors.append(f"Missing required fields: {missing_keys}")

    if extra_keys:
        warnings.append(f"Unexpected extra fields: {extra_keys}")

    for field_name in FIELD_ORDER:
        if field_name not in fields:
            continue

        field_value = fields[field_name]

        if not isinstance(field_value, dict):
            errors.append(f"{field_name}: field value must be an object.")
            continue

        value = field_value.get("value")
        sources = field_value.get("sources")
        confidence = field_value.get("confidence")
        notes = field_value.get("notes")

        if not isinstance(sources, list):
            errors.append(f"{field_name}: sources must be a list.")
            continue

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            errors.append(
                f"{field_name}: confidence must be a number between 0 and 1."
            )

        if value is None:
            null_count += 1
            if not isinstance(notes, str) or not notes.strip():
                errors.append(
                    f"{field_name}: null value must include an explanation in notes."
                )
            continue

        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"{field_name}: non-null value must be a non-empty string."
            )
            continue

        populated_count += 1

        if not sources:
            errors.append(
                f"{field_name}: non-null value has no source citations."
            )
            continue

        field_citations_valid = True

        for citation in sources:
            citation_count += 1

            if not isinstance(citation, dict):
                errors.append(
                    f"{field_name}: citation must be an object."
                )
                field_citations_valid = False
                continue

            file_name = citation.get("file")
            chunk_id = citation.get("chunk_id")
            quote = citation.get("quote")

            if not isinstance(file_name, str) or not file_name.strip():
                errors.append(
                    f"{field_name}: citation is missing its filename."
                )
                field_citations_valid = False

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                errors.append(
                    f"{field_name}: citation is missing its chunk_id."
                )
                field_citations_valid = False
            elif (
                known_chunk_ids is not None
                and chunk_id not in known_chunk_ids
            ):
                errors.append(
                    f"{field_name}: citation uses unknown chunk_id {chunk_id!r}."
                )
                field_citations_valid = False

            if not isinstance(quote, str) or not quote.strip():
                errors.append(
                    f"{field_name}: citation has no supporting quote."
                )
                field_citations_valid = False

        if field_citations_valid:
            valid_citation_count += len(sources)

    validation_summary = payload.get("validation_summary", {})

    if not isinstance(validation_summary, dict):
        errors.append("'validation_summary' must be a JSON object.")
        validation_summary = {}
    elif validation_summary.get("passed") is not True:
        errors.append(
            "The embedded validation_summary does not report passed=true."
        )

    if validation_summary.get("total_fields") != len(FIELD_ORDER):
        errors.append(
            f"Expected {len(FIELD_ORDER)} fields in validation_summary."
        )

    if null_count:
        warnings.append(
            f"{null_count} fields are null. Review whether each is genuinely "
            "unsupported by the source documents."
        )

    addendum_log = payload.get("addendum_change_log")

    if not isinstance(addendum_log, dict):
        errors.append("'addendum_change_log' must be an object.")
        addendum_log = {}

    changes = addendum_log.get("changes", [])

    if not isinstance(changes, list):
        errors.append("Addendum 'changes' must be a list.")
        changes = []

    return {
        "bid_id": bid_id,
        "file": str(output_path),
        "passed": len(errors) == 0,
        "total_required_fields": len(FIELD_ORDER),
        "actual_field_count": len(fields),
        "populated_fields": populated_count,
        "null_fields": null_count,
        "citation_count": citation_count,
        "citations_with_valid_metadata": valid_citation_count,
        "addendum_change_count": len(changes),
        "embedded_validation_passed": validation_summary.get("passed"),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quality-check final Bid1 and Bid2 JSON outputs."
    )
    parser.add_argument(
        "--bid-id",
        choices=["Bid1", "Bid2"],
        help="Check only one bid. Omit to check both.",
    )
    args = parser.parse_args()

    bid_ids = [args.bid_id] if args.bid_id else ["Bid1", "Bid2"]

    known_chunk_ids, chunk_note = load_known_chunk_ids()
    results = [
        check_bid(bid_id, known_chunk_ids)
        for bid_id in bid_ids
    ]

    if chunk_note:
        for result in results:
            result.setdefault("warnings", []).append(chunk_note)

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": results,
        "overall_passed": all(
            result.get("passed") is True for result in results
        ),
    }

    report_path = PROJECT_ROOT / "outputs" / "quality_check_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for result in results:
        print(f"\n--- {result['bid_id']} ---")
        print(f"File: {result['file']}")
        print(f"Passed: {result['passed']}")

        for key in [
            "total_required_fields",
            "actual_field_count",
            "populated_fields",
            "null_fields",
            "citation_count",
            "citations_with_valid_metadata",
            "addendum_change_count",
        ]:
            if key in result:
                print(f"{key}: {result[key]}")

        for error in result.get("errors", []):
            print(f"ERROR: {error}")

        for warning in result.get("warnings", []):
            print(f"WARNING: {warning}")

    print(f"\nOverall passed: {report['overall_passed']}")
    print(f"Report saved: {report_path}")

    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
