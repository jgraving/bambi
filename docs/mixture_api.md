# Finite-mixture prototype

This draft implements an initial subset of the API proposed in
[upstream issue 1040](https://github.com/bambinos/bambi/issues/1040) and
[issue 1041](https://github.com/bambinos/bambi/issues/1041):

```python
family = bmb.MixtureVonMises(n=2, order_by="Intercept")
formula = bmb.Formula(
    "y ~ mixture(mu, kappa)",
    "mu ~ x + (1 | animal)",
    "kappa ~ z",
    "weights ~ w + (1 | animal)",
)
model = bmb.Model(formula, data, family=family)
```

Parameters in `mixture(...)` have a component axis. Unindexed formulas use
the same predictors with separate coefficients for every component. Other
parameters remain shared and can still vary by observation. `weights` uses
softmax with a zero first logit; omitting its formula gives a Dirichlet prior.
Component labels are one-based. The latent component index is marginalized.

`Mixture(component, n)` accepts a scalar numeric family, including compatible
custom families, and retains its links and likelihood parameter transforms.
Custom distributions must provide a PyMC-compatible `.dist` constructor.
Families with registered response-data transforms are rejected explicitly:
applying those transforms inside a mixture is not supported by this prototype.
`MixtureVonMises` is a convenience constructor. This draft does not add the
other built-in constructors or automatic identifying priors proposed in 1040.
Indexed formulas (`mu[1] ~ x`), heterogeneous component families, nonlinear
formulas, response wrappers, and posterior class responsibilities remain out
of scope. Predicted `weights` are mixing probabilities conditional on
covariates, not probabilities conditional on the observed response.

## Circular component reporting

`order_by` is optional. It names a population coefficient with one value per
component, usually `Intercept`. For a von Mises family, the `mu` intercept
is wrapped to `[-pi, pi)` and sorted once; its centered and aliased forms are
also recognized. Slopes and other coefficients, such as `kappa_Intercept`,
retain their numerical order. Adding a full turn to a continuous-predictor
slope generally changes directions at intermediate predictor values.
The same indices create
`*_ordered` views of
all component parameters, coefficients, hyperparameters, and group effects.
Weight coefficients restore their fixed zero reference before permutation.
These views are full logits, not coefficients for a new reduced-logit model.
The reference component's restored weight hyperparameters are bookkeeping
zeros, not additional inferred scales. Original coefficient draws remain
available and are used for prediction.
Wrapping changes only the ordering key; ordered views preserve the original
parameter values rather than wrapping every angle-valued draw.

### Random effects and boundary crossings

For a population intercept $\alpha_k$, slopes $\beta_k$, and an angular random
effect $u_{ik}$, the predictor is
$\eta_{ik}=\alpha_k+x_i^\mathsf{T}\beta_k+u_{ik}$. The inverse link is applied
after adding all these contributions. A circular wrapping link therefore
produces $\mu_{ik}=\operatorname{wrap}(\eta_{ik})$, not a sum of independently
wrapped coefficients. With the default identity link, the von Mises density
still interprets this predictor periodically.

When `order_by="Intercept"`, every posterior draw uses the single permutation
$p=\operatorname{argsort}(\operatorname{wrap}(\alpha))$. The ordered direction,
random effect, concentration, and weight are then
$\mu_{i,p_k}$, $u_{i,p_k}$, $\kappa_{i,p_k}$, and $w_{i,p_k}$. Component-specific
random-effect hyperparameters and weight/concentration effects follow the same
permutation. Random effects are not sorted or recentered independently, and
the permutation is not recomputed for each animal.

An animal's random effect can carry a direction across the cut. For example,
population directions `[3.1, -1.0]` have permutation `[1, 0]`. Offsets `[0.1, 0]`
produce wrapped individual directions approximately `[-3.083, -1.0]`; under
the population ordering these are `[-1.0, -3.083]`. They are deliberately not
numerically ascending: each direction still carries the correct random effect,
concentration, and weight. Sorting each animal separately would define a
different reporting convention and could assign different component identities
across animals. A crossing of the population intercept itself can still change
the common ordering, as discussed below.

Aliases are supported for component parameters, coefficients, and weights;
renaming a parameter does not change the likelihood or the common permutation.
When the ordering coefficient itself has an alias, use that alias in
`order_by`. `component_order` records zero-based indices into the original
component axis; the one-based coordinates label positions in the ordered view.

The likelihood uses the aligned component parameters; its density is unchanged
by their simultaneous permutation. Sorting does not constrain the sampled
parameters or remove multimodality. It also does not remove the branch cut:
even with two components, a direction crossing the cut changes its numerical
rank. More components are supported algebraically; their statistical
identification is not guaranteed. Use scientifically justified identifying
structure or a suitable relabelling analysis, and inspect predictive density.

The [synthetic hierarchical example](notebooks/vonmises_mixture.py) keeps angular
effects centered while retaining non-centered Normal effects elsewhere.
The [simulation check](notebooks/mixture_ordering_validation.py) compares two,
three, and four components away from and near the branch cut, using known
generating identities solely to evaluate the relabelling convention.

See the [simulation results](validation/mixture_ordering.md) and
[comparison with bounded ordering](validation/ordered_comparison.md) for the
observed distinction between sampling geometry and stable physical labels.
