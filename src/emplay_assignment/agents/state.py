
from __future__ import annotations

from typing import Any, Literal, TypedDict


class BidAgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    # Input
    bid_id: str
    mode: Literal["extract", "qa"]
    question: str
    requested_fields: list[str]

    # Planning and retrieval
    plan: list[dict[str, Any]]
    evidence: list[dict[str, Any]]

    # Extraction and reconciliation
    draft_fields: dict[str, Any]
    addendum_changes: list[dict[str, Any]]

    # Validation and retry control
    validation_results: dict[str, Any]
    retry_count: int
    retry_limit: int

    # Outputs and observability
    answer: str
    final_output: dict[str, Any]
    errors: list[str]
    agent_trace: list[dict[str, Any]]
