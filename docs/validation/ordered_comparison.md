# Does an ordering constraint make boundary sampling harder?

- **Tested:** Unconstrained circular sampling with argsort reporting versus a
  bounded ordering constraint, using matched data and exchangeable priors.
- **Why:** A component crossing ±π changes numerical rank. Check whether an
  ordering constraint makes it harder for chains to explore both sides.
- **Result:** Away from the boundary, both methods had acceptable diagnostics.
  Near it, unconstrained fits had sorted R-hat below 1.008 and truth-matched
  bulk ESS above 2,200; bounded ordering gave R-hat around 1.74 and ESS around six.
- **Use:** These examples support unconstrained circular sampling with a common
  permutation for reporting near the boundary. Check mixing even with zero
  divergences, and assess physical labels separately. The sampling benefit is
  from the parametrization; argsort keeps associated parameters together.

![Panel A: a separate six-fit study finds that sorted ranks can change physical identity near the boundary. Panel B: for near-boundary K = 2, 3, and 4 fits, unconstrained circular sampling gives sorted R-hat of 1.007, 1.006, and 1.004, versus 1.743, 1.737, and 1.738 with bounded ordering.](mixture_ordering_overview.svg)

*Panel B: check chain mixing even when there are no divergences. Low sorted
R-hat does not establish fixed physical labels; panel A shows that distinction
in the [separate identity study](mixture_ordering.md).*

## Results

| K | Scenario | Method | Largest sorted R-hat | Minimum truth-matched bulk ESS | Posterior NUTS steps | Fit time including compilation (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2 | Away from cut | argsort | 1.002 | 2272 | 11,120 | 30.5 |
| 2 | Away from cut | ordered | 1.006 | 1087 | 22,284 | 21.9 |
| 2 | Near cut | argsort | 1.007 | 2206 | 10,966 | 9.3 |
| 2 | Near cut | ordered | 1.743 | 6 | 314,582 | 79.0 |
| 3 | Near cut | argsort | 1.006 | 2612 | 13,514 | 15.9 |
| 3 | Near cut | ordered | 1.737 | 6 | 412,212 | 125.6 |
| 4 | Near cut | argsort | 1.004 | 2406 | 14,752 | 13.8 |
| 4 | Near cut | ordered | 1.738 | 6 | 466,536 | 173.3 |

All eight runs had zero divergences. Near-cut diagnostics are consistent with
ordered chains remaining in distinct boundary representations. Roughly half
their samples occupied each representation, reflecting balanced initialization;
these are unreliable posterior probabilities because the chains did not converge.
Unconstrained runs explored the much smaller cross-cut mass. The resulting
rank changes are expected when a physical component crosses the boundary.

Truth matching uses simulated generating directions as an oracle and is
unavailable for real observations. Sorted R-hat assesses rank-defined
parameters and does not guarantee stable physical identities. The
[initial simulations](mixture_ordering.md) measure this distinction directly.

## How it was tested

Both methods use the same data and exchangeable priors as the
[initial simulations](mixture_ordering.md). The constrained model replaces
PyMC's circular transform with a chain of an interval transform and the
standard ordered transform:

```python
pm.VonMises(
    "mu", mu=0, kappa=1e-6, dims="component",
    default_transform=pm.distributions.transforms.Chain([
        pm.distributions.transforms.Interval(-np.pi, np.pi),
        pm.distributions.transforms.Ordered(),
    ]),
)
```

In the backward direction, ordered real values pass through a monotone
bounded map, so the resulting directions are ordered and inside the angular
interval. The script asserts both properties on every retained draw. This
avoids a hard support boundary as a confounder.

Simply passing `transform=ordered` to a von Mises prior in PyMC 6.3.2 adds the
ordered transform to the default circular transform. In the backward direction
it orders first, then wraps. For example, ordered real values `0` and `pi + 0.1`
wrap to `0` and `-pi + 0.1`, which are not ordered. That combination is not the
true ordering constraint tested here.

Because the complete joint prior is exchangeable across components, the
posterior restricted to ordered means equals the common-permutation argsort
pushforward up to the constant K! normalizer. Concentrations and weights must
be permuted together with the means for this equivalence. This argument does
not apply unchanged to non-exchangeable component priors.

Each fit used four chains, 750 tuning iterations, 500 retained draws per chain,
and 600 observations. For the near-cut cases, two chains started with the
boundary direction just below pi and two just above -pi. Ordered chains were
initialized by sorting the directions and the associated parameters together.
This deliberately tests both circular representations instead of allowing all
chains to remain in the same representation unnoticed.

## What this does not establish

These well-separated synthetic mixtures do not establish a general performance
advantage. Data, priors, samplers, tuning, and initialization can change the
results. Fit times include compilation in one local run; NUTS steps depend less
on the environment. Hierarchical models need the same permutation applied to
associated random effects, but their sampling performance was not tested here.

## Reproduce

```bash
python docs/notebooks/compare_mixture_ordering.py --output results
```

The script writes JSON only. After rerunning, review the new results and manually
update this report's summary and table. Archived inputs and outputs are in
[mixture_ordering_comparison.json](mixture_ordering_comparison.json).

Refresh the figure from saved JSON without fitting:

```bash
python docs/notebooks/plot_mixture_ordering.py
```

Use `--data-dir results` for a new run containing both experiments' JSON files.
