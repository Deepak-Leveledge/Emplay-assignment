
from __future__ import annotations

import argparse
import json
import logging

from emplay_assignment.agents.qa_graph import qa_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph RFP Q&A workflow."
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the bid documents.",
    )

    parser.add_argument(
        "--bid-id",
        choices=["Bid1", "Bid2"],
        default=None,
        help="Optional bid filter.",
    )

    args = parser.parse_args()

    result = qa_graph.invoke({
        "question": args.question,
        "bid_id": args.bid_id,
        "mode": "qa",
        "plan": [],
        "evidence": [],
        "errors": [],
        "agent_trace": [],
    })

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result.get("answer", "No answer returned."))

    print("\nSOURCES")
    print("-" * 70)

    for source in result.get("final_output", {}).get("sources", []):
        print(
            f"[{source.get('source_id')}] "
            f"{source.get('file_name')} | "
            f"page={source.get('page_number')} | "
            f"chunk={source.get('chunk_id')}"
        )

    print("\nAGENT TRACE")
    print("-" * 70)
    print(
        json.dumps(
            result.get("agent_trace", []),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    if result.get("errors"):
        print("\nERRORS")
        print(json.dumps(result["errors"], indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
