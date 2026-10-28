"""Test numerical ordering against known component identities in simulations.

Run with ``python docs/notebooks/mixture_ordering_validation.py --output results``.
This uses direct PyMC models to isolate the relabelling method from Bambi's API.
Truth matching is an evaluation oracle available only for simulated data, not
a proposed inference method. The paired scenarios differ only by a rotation.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import arviz_stats as azs
import numpy as np
import pymc as pm
from scipy.optimize import linear_sum_assignment
from scipy.special import i0e, logsumexp


def wrap(value):
    """Return the principal angular difference in radians."""
    return (value + np.pi) % (2 * np.pi) - np.pi


def mixture_logp(mu, kappa, weights, grid):
    """Evaluate the marginalized circular density for each posterior draw."""
    component = (
        kappa[..., None, :] * np.cos(grid[:, None] - mu[..., None, :])
        - np.log(2 * np.pi)
        - np.log(i0e(kappa[..., None, :]))
        - kappa[..., None, :]
    )
    return logsumexp(np.log(weights[..., None, :]) + component, axis=-1)


def evaluate(n, boundary, draws, tune, observations, method="argsort", balanced_branch_init=False):
    """Fit a simulation and compare argsort with truth-matched identities."""
    seed = 1900 + n
    rng = np.random.default_rng(seed)
    means = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
    kappas = np.linspace(10, 18, n)
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    labels = rng.choice(n, observations, p=weights)
    y = rng.vonmises(means[labels], kappas[labels])
    if boundary:
        rotation = np.pi - 0.02 - means[-1]
        means, y = wrap(means + rotation), wrap(y + rotation)
    # Different label permutations across chains deliberately exercise relabelling.
    initial = []
    for chain in range(4):
        initial_means = means.copy()
        if boundary and balanced_branch_init and chain % 2:
            initial_means[-1] = -np.pi + 0.02
        permutation = (
            np.argsort(initial_means) if method == "ordered" else np.roll(np.arange(n), chain % n)
        )
        initial.append(
            {
                "mu": initial_means[permutation],
                "kappa": kappas[permutation],
                "weights": weights[permutation],
            }
        )
    with pm.Model(coords={"component": range(n)}) as model:
        options = {}
        if method == "ordered":
            # Replace the circular transform. Merely adding transform=ordered
            # chains it with circular wrapping, which can undo the ordering.
            # Backward: ordered unconstrained values, then a monotone bounded map.
            options["default_transform"] = pm.distributions.transforms.Chain(
                [
                    pm.distributions.transforms.Interval(-np.pi, np.pi),
                    pm.distributions.transforms.Ordered(),
                ]
            )
        elif method != "argsort":
            raise ValueError(f"Unknown method: {method}")
        mu = pm.VonMises("mu", mu=0, kappa=1e-6, dims="component", **options)
        kappa = pm.LogNormal("kappa", mu=np.log(12), sigma=0.7, dims="component")
        w = pm.Dirichlet("weights", a=np.ones(n), dims="component")
        pm.Mixture("y", w=w, comp_dists=pm.VonMises.dist(mu=mu, kappa=kappa), observed=y)
        started = perf_counter()
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            cores=1,
            init="adapt_diag",
            initvals=initial,
            target_accept=0.9,
            random_seed=seed,
            progressbar=False,
        )
        fit_seconds = perf_counter() - started
    posterior = idata.posterior
    raw = {name: posterior[name].values for name in ("mu", "kappa", "weights")}
    if method == "ordered":
        assert np.all(np.diff(raw["mu"], axis=-1) > 0)
        assert np.all(np.abs(raw["mu"]) < np.pi)
    order = np.argsort(raw["mu"], axis=-1)
    ordered = {name: np.take_along_axis(value, order, axis=-1) for name, value in raw.items()}
    truth_order = np.argsort(means)
    matched = {name: np.empty_like(value) for name, value in raw.items()}
    disagree = []
    for chain in range(4):
        for draw in range(draws):
            cost = np.abs(wrap(raw["mu"][chain, draw, :, None] - means[None, :]))
            rows, columns = linear_sum_assignment(cost)
            permutation = rows[np.argsort(columns)]
            for name in raw:
                matched[name][chain, draw] = raw[name][chain, draw, permutation]
            labels_by_raw_index = np.empty(n, dtype=int)
            labels_by_raw_index[rows] = columns
            disagree.append(np.any(labels_by_raw_index[order[chain, draw]] != truth_order))
    oracle_errors = wrap(matched["mu"] - means)
    ordered_errors = wrap(ordered["mu"] - means[truth_order])
    raw_diagnostics = np.concatenate(list(raw.values()), axis=-1)
    ordered_diagnostics = np.concatenate(list(ordered.values()), axis=-1)
    matched_diagnostics = np.concatenate(
        [oracle_errors, matched["kappa"], matched["weights"]], axis=-1
    )
    grid = np.linspace(-np.pi, np.pi, 51)
    density_error = np.max(
        np.abs(mixture_logp(**raw, grid=grid) - mixture_logp(**ordered, grid=grid))
    )
    result = {
        "method": method,
        "balanced_branch_initialization": balanced_branch_init,
        "components": n,
        "scenario": "near_branch_cut" if boundary else "away_from_branch_cut",
        "observations": observations,
        "chains": 4,
        "draws_per_chain": draws,
        "tune_per_chain": tune,
        "seed": seed,
        "true_mu": means.tolist(),
        "true_kappa": kappas.tolist(),
        "true_weights": weights.tolist(),
        "max_log_density_difference_after_sorting": float(density_error),
        "fraction_draws_with_rank_identity_change": float(np.mean(disagree)),
        "angular_rmse_sorted_radians": float(np.sqrt(np.mean(ordered_errors**2))),
        "angular_rmse_truth_matched_radians": float(np.sqrt(np.mean(oracle_errors**2))),
        "weight_rmse_sorted": float(
            np.sqrt(np.mean((ordered["weights"] - weights[truth_order]) ** 2))
        ),
        "weight_rmse_truth_matched": float(np.sqrt(np.mean((matched["weights"] - weights) ** 2))),
        "max_rhat_raw": float(np.max(azs.rhat(raw_diagnostics))),
        "max_rhat_sorted": float(np.max(azs.rhat(ordered_diagnostics))),
        "max_rhat_truth_matched": float(np.max(azs.rhat(matched_diagnostics))),
        "min_bulk_ess_truth_matched": float(np.min(azs.ess(matched_diagnostics))),
        "divergences": int(idata.sample_stats.diverging.sum()),
        "fit_seconds_including_compilation": fit_seconds,
        "total_nuts_steps": int(idata.sample_stats.n_steps.sum()),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=750)
    parser.add_argument("--observations", type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for n in (2, 3, 4):
        for boundary in (False, True):
            print(f"Starting K={n}, near branch cut={boundary}", flush=True)
            results.append(evaluate(n, boundary, args.draws, args.tune, args.observations))
            (args.output / "mixture_ordering_results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
