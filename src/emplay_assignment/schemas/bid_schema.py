
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


REQUIRED_BID_FIELDS = [
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


class SourceCitation(BaseModel):
    """Reference to evidence retrieved from a source document."""

    model_config = ConfigDict(extra="forbid")

    file: str
    page: int | None = None
    chunk_id: str | None = None
    quote: str | None = None


class FieldValue(BaseModel):
    """A single extracted field with evidence and confidence."""

    model_config = ConfigDict(extra="forbid")

    value: Any = None
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "FieldValue":
        if self.value is not None and not self.sources:
            raise ValueError(
                "A non-null field value must have at least one source."
            )

        if self.value is None and not (
            self.notes and self.notes.strip()
        ):
            raise ValueError(
                "A null field value must include a reason in notes."
            )

        return self


class AddendumChange(BaseModel):
    """Records a source-supported change to a bid field."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    original_value: Any = None
    updated_value: Any = None
    source: SourceCitation
    explanation: str


class ValidationSummary(BaseModel):
    """Counts the outcomes of final field validation."""

    model_config = ConfigDict(extra="forbid")

    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    not_found: int = Field(default=0, ge=0)


class ExtractedBid(BaseModel):
    """Final machine-readable record for one bid."""

    model_config = ConfigDict(extra="forbid")

    bid_id: str
    fields: dict[str, FieldValue]
    addendum_changes: list[AddendumChange] = Field(
        default_factory=list
    )
    validation: ValidationSummary = Field(
        default_factory=ValidationSummary
    )

    @model_validator(mode="after")
    def validate_required_fields(self) -> "ExtractedBid":
        missing_fields = set(REQUIRED_BID_FIELDS) - set(self.fields)
        unknown_fields = set(self.fields) - set(REQUIRED_BID_FIELDS)

        if missing_fields:
            raise ValueError(
                f"Missing required fields: {sorted(missing_fields)}"
            )

        if unknown_fields:
            raise ValueError(
                f"Unexpected fields: {sorted(unknown_fields)}"
            )

        return self
