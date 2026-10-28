# Argsort reporting versus a bounded ordering constraint

In this controlled comparison, unconstrained circular sampling followed by a
common argsort permutation explored the near-boundary posterior much better
than an ordering constraint. The benefit comes from the unconstrained circular
parametrization. Argsort itself leaves the density unchanged and serves only
to align reported parameters. Neither strategy guarantees stable physical
labels at the branch cut.

## Comparable models

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

## Design and results

Each fit used four chains, 750 tuning iterations, 500 retained draws per chain,
and 600 observations. For the near-cut cases, two chains started with the
boundary direction just below pi and two just above -pi. Ordered chains were
initialized by sorting the directions and the associated parameters together.
This deliberately tests both circular representations instead of allowing all
chains to remain in the same representation unnoticed.

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

All eight runs had zero divergences. Away from the cut, both parametrizations
produced acceptable diagnostics in this example. Near the cut, bounded
ordering produced R-hat around 1.74 and an effective sample size around six,
consistent with chains remaining in distinct boundary representations.
Unconstrained circular sampling gave sorted R-hat below 1.008 and truth-matched
bulk ESS above 2,200. Zero divergences did not imply that the ordered chains
mixed adequately.

The ordered runs retained roughly half their samples in each boundary
representation, reflecting the deliberately balanced initialization. Those
frequencies are not reliable posterior probability estimates because the
chains did not converge. The unconstrained runs explored the much smaller
cross-cut mass without fixing physical component identity to numerical rank.
That exploration creates rank changes; it is not itself an inference failure.

The truth-matched diagnostic uses simulated generating directions as an oracle
and cannot be obtained this way for real observations. Sorted R-hat concerns
rank-defined parameters and likewise does not guarantee physical identity.

## Scope

These are specific, well-separated synthetic mixtures, not a general benchmark
or proof of superiority. More observations, overlapping components, different
priors, hierarchical effects, samplers, tuning, and initialization can change
the results. Fit time includes compilation and reflects one local run; NUTS
step counts provide a less environment-dependent indication of work. The
comparison should motivate further validation, not a universal performance claim.

Run `python docs/notebooks/compare_mixture_ordering.py --output results` to
reproduce. Complete inputs and outputs are in
[mixture_ordering_comparison.json](mixture_ordering_comparison.json).
