# Testing circular argsort reporting

The common-permutation method is not mathematically limited to two components.
In these simulations it preserved the mixture density for two, three, and four
components, and produced stable physical labels away from the angular cut.
Near the cut, numerical rank changed physical identity even with two components.

## Results

| Components | Scenario | Draws with a rank/identity mismatch | Angular RMSE by sorted rank (rad) | Angular RMSE after truth matching (rad) | Largest sorted R-hat |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | Away from cut | 0.00% | 0.023 | 0.023 | 1.002 |
| 2 | Near cut | 1.10% | 0.328 | 0.023 | 1.004 |
| 3 | Away from cut | 0.00% | 0.033 | 0.033 | 1.005 |
| 3 | Near cut | 12.50% | 0.744 | 0.033 | 1.004 |
| 4 | Away from cut | 0.00% | 0.038 | 0.038 | 1.007 |
| 4 | Near cut | 23.15% | 0.752 | 0.037 | 1.003 |

All six fits had zero divergences. The maximum log-density difference after
simultaneously permuting means, concentrations, and weights was below 1e-15.
Truth-matched maximum R-hat was below 1.008, and minimum bulk ESS exceeded 1,900.
Raw labels had high R-hat because chains deliberately started in different
permutations. Sorted R-hat was small even when sorted ranks mixed physical
identities: good diagnostics of rank-defined quantities do not establish
stable physical component labels.

## Method

Each case used 600 observations, four chains, 750 tuning iterations and 500
retained draws per chain. Components were well separated with concentrations
between 10 and 18 and unequal weights proportional to 1, ..., K. Priors were
exchangeable: von Mises(0, 1e-6) directions, LogNormal(log(12), 0.7)
concentrations, and Dirichlet(1, ..., 1) weights. Initial values used
permutations of generating parameters to test cross-chain relabelling.

For each K, the second dataset and means were a rotation of the first,
placing one direction 0.02 radians below pi. This holds component geometry and observations fixed relative to each other
while moving them relative to the branch cut. The direction prior is nearly,
but not exactly, uniform.

For evaluation only, each draw was matched to known generating directions by
minimum total circular distance (linear assignment). A rank/identity mismatch
means sorted ranks did not correspond to the generating directions' fixed
numerical order. Every associated parameter used the same permutation.
Truth matching is an evaluation oracle available only because these are
simulations; it is not an identification method for real data.

## Limits

These six cases use one dataset per K, separated components, and favorable
initial values. They do not establish sampler reliability, test overlapping
components or hierarchical models, or estimate general failure rates. The
increasing mismatch percentages are specific to these datasets and posterior
uncertainties, not evidence of a universal dependence on K. The relevant
failure is a direction's posterior crossing the chosen branch cut.

The simulations fit unsorted marginalized PyMC mixtures, then apply the same
argsort permutation to every component parameter. A common permutation leaves
the likelihood invariant and cannot improve its sampling geometry. Bambi's
unit tests separately verify the common permutation inside the likelihood,
alignment of hierarchical parameters, and new-data predictions.

## Reproduce

```bash
python docs/notebooks/mixture_ordering_validation.py --output results
```

This run used Python 3.12.14, PyMC 6.3.2, PyTensor 3.3.4, NumPy 2.5.4,
SciPy 1.17.1, and ArviZ Stats 1.3.3. Seeds and complete numerical results are in
[mixture_ordering_results.json](mixture_ordering_results.json).
