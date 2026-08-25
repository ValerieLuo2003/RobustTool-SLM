#!/usr/bin/env python
"""Create one validated multi-model robustness comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.eval.robustness_comparison import (
    compare_robustness_runs,
    write_robustness_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=run_directory")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    runs: list[tuple[str, Path]] = []
    for specification in args.run:
        if "=" not in specification:
            parser.error(f"--run must use LABEL=path: {specification}")
        label, raw_path = specification.split("=", 1)
        runs.append((label, Path(raw_path)))
    try:
        payload = compare_robustness_runs(runs)
        outputs = write_robustness_comparison(payload, args.output_prefix)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {**outputs, "run_count": len(runs), "pair_count": payload["pair_count"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
