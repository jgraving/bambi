"""Check whether bounded ordering impedes sampling near the angular boundary.

Compare unconstrained circular sampling with argsort reporting against a bounded
ordering constraint. Matched data and exchangeable priors isolate the effect of
parametrization; both target the same ordered posterior when every component
parameter follows the same permutation (up to the normalizing factor K!).

Archived results: unconstrained fits mixed better near the boundary, although
all runs had zero divergences. Check chain mixing separately from physical
labels. Details: docs/validation/ordered_comparison.md.

Run ``python docs/notebooks/compare_mixture_ordering.py --output results``.
This writes JSON only; review new results and refresh the report manually.
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
