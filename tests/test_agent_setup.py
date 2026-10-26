
from emplay_assignment.agents.state import BidAgentState
from emplay_assignment.agents.search_tools import search_bid_documents


def test_shared_state_contains_required_fields():
    annotations = BidAgentState.__annotations__

    expected_fields = {
        "bid_id",
        "mode",
        "plan",
        "evidence",
        "draft_fields",
        "addendum_changes",
        "validation_results",
        "retry_count",
        "final_output",
        "agent_trace",
    }

    assert expected_fields.issubset(set(annotations))


def test_search_tool_name():
    assert search_bid_documents.name == "search_bid_documents"


def test_search_tool_has_validated_arguments():
    schema = search_bid_documents.args_schema

    assert schema is not None

    fields = schema.model_fields

    assert {"query", "bid_id", "top_k"}.issubset(set(fields))


def test_search_tool_schema_limits_top_k():
    schema = search_bid_documents.args_schema
    top_k_field = schema.model_fields["top_k"]

    assert top_k_field.default == 5
    assert top_k_field.annotation is int
