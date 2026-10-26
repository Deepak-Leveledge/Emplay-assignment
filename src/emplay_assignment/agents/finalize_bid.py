
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from emplay_assignment.agents.validator_critic import (
    FIELD_ORDER,
    load_known_chunk_ids,
    validate_final_fields,
)
from emplay_assignment.config import PROJECT_ROOT


AGENT_OUTPUTS = [
    ("logistics", "logistics"),
    ("commercial_legal", "commercial_legal"),
    ("product_specs", "product_specifications"),
    ("identity_summary", "identity_summary"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required output file is missing: {path}\n"
            "Run the corresponding agent first, then retry final assembly."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return payload


def assemble_final_bid(bid_id: str) -> dict[str, Any]:
    if bid_id not in {"Bid1", "Bid2"}:
        raise ValueError("bid_id must be either 'Bid1' or 'Bid2'")

    output_dir = PROJECT_ROOT / "outputs"
    merged_fields: dict[str, Any] = {}
    merge_errors: list[str] = []
    agent_runs: dict[str, Any] = {}

    # Load and merge the four extraction groups.
    for file_suffix, group_name in AGENT_OUTPUTS:
        path = output_dir / f"{bid_id}_{file_suffix}.json"
        payload = read_json(path)

        if payload.get("bid_id") != bid_id:
            merge_errors.append(
                f"{path.name} belongs to bid {payload.get('bid_id')!r}, "
                f"not {bid_id!r}."
            )

        agent_fields = payload.get("fields")

        if not isinstance(agent_fields, dict):
            raise ValueError(
                f"{path.name} must contain a 'fields' object."
            )

        for field_name, field_value in agent_fields.items():
            if field_name in merged_fields:
                merge_errors.append(
                    f"Duplicate field '{field_name}' found across agent outputs."
                )
                # Keep the first result and report the duplicate for review.
                continue

            merged_fields[field_name] = field_value

        agent_runs[group_name] = {
            "output_file": path.name,
            "evidence_count": payload.get("evidence_count"),
            "latency_ms": payload.get("latency_ms"),
            "model": payload.get("model"),
        }

    # Add a clear null object for any missing required field rather than
    # silently omitting that key from the final JSON.
    for field_name in FIELD_ORDER:
        if field_name not in merged_fields:
            merged_fields[field_name] = {
                "value": None,
                "sources": [],
                "confidence": 0.0,
                "notes": (
                    "No extraction result was available for this required field."
                ),
            }

    # Keep exact assignment field names and deterministic ordering.
    ordered_fields = {
        field_name: merged_fields[field_name]
        for field_name in FIELD_ORDER
    }

    addendum_path = output_dir / f"{bid_id}_addendum_changes.json"
    addendum_payload = read_json(addendum_path)

    if addendum_payload.get("bid_id") != bid_id:
        merge_errors.append(
            f"{addendum_path.name} belongs to bid "
            f"{addendum_payload.get('bid_id')!r}, not {bid_id!r}."
        )

    known_chunk_ids, chunk_check_note = load_known_chunk_ids()

    validation_summary = validate_final_fields(
        fields=ordered_fields,
        addendum_payload=addendum_payload,
        known_chunk_ids=known_chunk_ids,
        chunk_check_note=chunk_check_note,
        inherited_errors=merge_errors,
    )

    addendum_change_log = {
        "changes": addendum_payload.get("changes", []),
        "review_notes": addendum_payload.get("review_notes", []),
        "addenda_found_in_retrieved_evidence": addendum_payload.get(
            "addenda_found_in_retrieved_evidence", []
        ),
        "evidence_count": addendum_payload.get("evidence_count"),
        "latency_ms": addendum_payload.get("latency_ms"),
        "scope_note": (
            "This change log reflects retrieved passages and must not be "
            "treated as proof that every document and every addendum was "
            "exhaustively reconciled."
        ),
    }

    agent_runs["addendum_reconciliation"] = {
        "output_file": addendum_path.name,
        "evidence_count": addendum_payload.get("evidence_count"),
        "latency_ms": addendum_payload.get("latency_ms"),
        "model": addendum_payload.get("model"),
    }

    return {
        "schema_version": "1.0",
        "bid_id": bid_id,
        "fields": ordered_fields,
        "addendum_change_log": addendum_change_log,
        "validation_summary": validation_summary,
        "agent_runs": agent_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and assemble the final RFP bid JSON."
    )
    parser.add_argument(
        "--bid-id",
        choices=["Bid1", "Bid2"],
        required=True,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path.",
    )
    args = parser.parse_args()

    result = assemble_final_bid(args.bid_id)

    output_path = Path(
        args.output or f"outputs/{args.bid_id}_final.json"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = result["validation_summary"]

    print(f"\nFinal JSON saved: {output_path}")
    print(f"Bid ID: {result['bid_id']}")
    print(f"Fields: {summary['total_fields']}")
    print(f"Populated fields: {summary['populated_fields_count']}")
    print(f"Null/missing fields: {summary['missing_fields_count']}")
    print(f"Citation coverage: {summary['citation_coverage']:.1%}")
    print(f"Addendum changes: {summary['addendum_changes_count']}")
    print(f"Validation passed: {summary['passed']}")

    if summary["fields_missing"]:
        print("\nFields requiring review:")
        for field_name in summary["fields_missing"]:
            print(f"  - {field_name}")

    if summary["errors"]:
        print("\nValidation errors:")
        for error in summary["errors"]:
            print(f"  - {error}")

    if summary["warnings"]:
        print("\nWarnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
