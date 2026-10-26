
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from emplay_assignment.agents.logistics_agent import extract_logistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract RFP dates and logistics fields."
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

    result = extract_logistics(args.bid_id)

    serialized = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )

    print(serialized)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
