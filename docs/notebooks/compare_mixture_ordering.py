"""Compare argsort reporting with a bounded ordered angular parametrization.

Run ``python docs/notebooks/compare_mixture_ordering.py --output results``.
Exchangeable priors make the joint ordered posterior equal to the common
argsort pushforward up to the normalizing factor K!. Differences in exploration
therefore reflect the sampling parametrization, not different intended priors.
This equivalence requires permuting all associated parameters together.
"""

import argparse
import json
from pathlib import Path

from mixture_ordering_validation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=750)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for n, boundary in ((2, False), (2, True), (3, True), (4, True)):
        for method in ("argsort", "ordered"):
            print(
                f"Starting K={n}, near branch cut={boundary}, method={method}",
                flush=True,
            )
            results.append(
                evaluate(
                    n,
                    boundary,
                    args.draws,
                    args.tune,
                    600,
                    method=method,
                    balanced_branch_init=True,
                )
            )
            (args.output / "mixture_ordering_comparison.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
