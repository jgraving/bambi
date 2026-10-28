"""Compare wrapping, sampling, and diagnostics in circular regression.

Run from the PR branch with::

    python docs/notebooks/wrap_validation.py --output docs/validation/wrap

Each identity/wrap pair uses identical data, explicit priors, starting points,
and random seeds. These deliberately small models isolate representation from
sampling. Deliberately separated periodic modes are not identified by wrapping.
The numerical stress test additionally requires mpmath (``pip install mpmath``).
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import arviz_stats as azs
import matplotlib
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

import bambi as bmb
from bambi.backend.pymc.links import wrap as symbolic_wrap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

SCENARIOS = (
    "crossing_regression",
    "intercept_aliases",
    "at_branch_cut",
    "slope_aliases",
)


def wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def diagnostics(values):
    """Rank-normalized split R-hat and bulk ESS, retaining every coordinate."""
    return {
        "rhat": np.asarray(azs.rhat(values)).tolist(),
        "bulk_ess": np.asarray(azs.ess(values)).tolist(),
    }


def scenario_spec(scenario, seed):
    rng = np.random.default_rng(seed)
    kappa = 8.0
    if scenario == "crossing_regression":
        x = np.linspace(-3, 3, 160)
        truth = 0.4 + 2 * x
        formula = "y ~ x"
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0.4, sigma=0.5),
            "x": bmb.Prior("Normal", mu=2, sigma=0.5),
        }
        initial = [{"Intercept": 0.4 + d, "x": 2 - d} for d in (-0.1, 0.1, -0.2, 0.2)]
        probes = np.linspace(-3, 3, 13)
        y = rng.vonmises(truth, kappa)
    elif scenario == "intercept_aliases":
        x = np.zeros(160)
        formula = "y ~ 1"
        priors = {"Intercept": bmb.Prior("Normal", mu=0, sigma=20)}
        initial = [{"Intercept": 0.6 + 2 * np.pi * turn} for turn in (-1, 0, 1, 2)]
        probes = np.array([0.0])
        y = rng.vonmises(0.6, kappa, len(x))
    elif scenario == "at_branch_cut":
        x = np.zeros(160)
        formula = "y ~ 1"
        priors = {"Intercept": bmb.Prior("Normal", mu=np.pi, sigma=0.5)}
        initial = [{"Intercept": np.pi + d} for d in (-0.1, 0.1, -0.2, 0.2)]
        probes = np.array([0.0])
        # Paired errors put the sample mean direction exactly on the branch cut.
        errors = rng.vonmises(0, kappa, len(x) // 2)
        y = wrap(np.pi + np.concatenate([errors, -errors]))
    elif scenario == "slope_aliases":
        x = np.tile([0.0, 1.0], 80)
        formula = "y ~ 0 + x"
        priors = {"x": bmb.Prior("Normal", mu=0, sigma=20)}
        initial = [{"x": 1 + 2 * np.pi * turn} for turn in (-1, 0, 1, 2)]
        probes = np.array([1.0, 0.5])
        y = rng.vonmises(x, kappa)
    else:
        raise ValueError(scenario)
    priors["kappa"] = kappa
    return pd.DataFrame({"x": x, "y": y}), formula, priors, initial, probes


def build(scenario, seed, link):
    data, formula, priors, initial, probes = scenario_spec(scenario, seed)
    model = bmb.Model(
        formula,
        data,
        family="vonmises",
        link=link,
        priors=priors,
        center_predictors=False,
    )
    model.build()
    return model, initial, probes


def fit_one(scenario, seed, link, draws, tune):
    model, initial, probes = build(scenario, seed, link)
    started = perf_counter()
    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=4,
        cores=1,
        init="adapt_diag",
        initvals=initial,
        target_accept=0.9,
        random_seed=seed + 10000,
        progressbar=False,
        compute_convergence_checks=False,
    )
    elapsed = perf_counter() - started
    posterior = idata.posterior
    if scenario == "crossing_regression":
        eta = posterior.Intercept.values[..., None] + posterior.x.values[..., None] * probes
        coefficients = np.stack([posterior.Intercept.values, posterior.x.values], axis=-1)
    elif scenario == "slope_aliases":
        eta = posterior.x.values[..., None] * probes
        coefficients = posterior.x.values[..., None]
    else:
        eta = posterior.Intercept.values[..., None]
        coefficients = eta.copy()
    angles = wrap(eta)
    unit = np.concatenate([np.sin(eta), np.cos(eta)], axis=-1)
    center = np.arctan2(np.sin(eta).mean(axis=(0, 1)), np.cos(eta).mean(axis=(0, 1)))
    residual = wrap(eta - center)
    prediction = model.predict(idata, data=pd.DataFrame({"x": probes}), inplace=False)
    reported = prediction.predictions.mu.values
    expected = angles if link == "wrap" else eta
    np.testing.assert_allclose(reported, expected, atol=1e-12)
    stats = idata.sample_stats
    turns = np.floor((coefficients + np.pi) / (2 * np.pi))
    result = {
        "scenario": scenario,
        "link": link,
        "seed": seed,
        "chains": 4,
        "draws_per_chain": draws,
        "tune_per_chain": tune,
        "probe_x": probes.tolist(),
        "coefficients": diagnostics(coefficients),
        "reported_mu": diagnostics(reported),
        "posthoc_wrapped_mu": diagnostics(angles),
        "sin_cos_mu": diagnostics(unit),
        "centered_angular_residual": diagnostics(residual),
        "chain_coefficient_means": coefficients.mean(axis=1).tolist(),
        "chain_direction_means": np.arctan2(
            np.sin(eta).mean(axis=1), np.cos(eta).mean(axis=1)
        ).tolist(),
        "pooled_direction_mean": center.tolist(),
        "pooled_wrapped_arithmetic_mean": angles.mean(axis=(0, 1)).tolist(),
        "coefficient_period_crossings": int(np.count_nonzero(np.diff(turns, axis=1))),
        "divergences": int(stats.diverging.sum()),
        "total_nuts_steps": int(stats.n_steps.sum()),
        "max_tree_depth": int(stats.tree_depth.max()),
        "fit_seconds_including_compilation": elapsed,
    }
    trace = {"eta": eta, "wrapped": angles, "coefficients": coefficients}
    return result, trace


def graph_equivalence():
    """Compare actual Bambi joint densities and gradients at identical points."""
    results = []
    for scenario in SCENARIOS:
        models = [build(scenario, 2400, link)[0] for link in ("identity", "wrap")]
        graphs = [m.backend.model for m in models]
        logp = [m.compile_logp() for m in graphs]
        grad = [m.compile_dlogp() for m in graphs]
        initial = scenario_spec(scenario, 2400)[3]
        errors = []
        for values in initial:
            for shift in (-1e-6, 0.0, 1e-6):
                point = {name: np.asarray(value + shift) for name, value in values.items()}
                errors.append(
                    [
                        abs(float(logp[0](point) - logp[1](point))),
                        float(np.max(np.abs(grad[0](point) - grad[1](point)))),
                    ]
                )
        results.append(
            {
                "scenario": scenario,
                "max_abs_joint_logp_difference": max(e[0] for e in errors),
                "max_abs_joint_gradient_difference": max(e[1] for e in errors),
            }
        )
    return results


def numerical_stress():
    """Stress argument reduction; these enormous predictors are not realistic fits."""
    import mpmath as mp  # pylint: disable=import-outside-toplevel

    mp.mp.dps = 80
    eta = pt.dvector("eta")
    # Subtracting the observed angle and adding pi both incur rounding at large eta.
    logps = [
        pm.logp(pm.VonMises.dist(mu=mu, kappa=np.float64(8)), np.float64(0.4))
        for mu in (eta, symbolic_wrap(eta))
    ]
    evaluate = pytensor.function([eta], [*logps, *[pt.grad(v.sum(), eta) for v in logps]])
    values = np.array([0.1, np.pi, 10, 1e3, 1e6, 1e9, 1e12, 1e15, 1e16])
    actual = evaluate(values)
    rows = []
    for i, value in enumerate(values):
        exact = mp.mpf(float(value))
        difference = exact - mp.mpf(0.4)
        reference = 8 * mp.cos(difference) - mp.log(2 * mp.pi) - mp.log(mp.besseli(0, 8))
        reference_gradient = -8 * mp.sin(difference)
        rows.append(
            {
                "eta": float(value),
                "identity_logp_absolute_error": float(abs(mp.mpf(actual[0][i]) - reference)),
                "wrap_logp_absolute_error": float(abs(mp.mpf(actual[1][i]) - reference)),
                "identity_gradient_absolute_error": float(
                    abs(mp.mpf(actual[2][i]) - reference_gradient)
                ),
                "wrap_gradient_absolute_error": float(
                    abs(mp.mpf(actual[3][i]) - reference_gradient)
                ),
            }
        )
    return rows


def representation_control(seed=789):
    """IID control: exactly equivalent angular draws stored in different periods."""
    rng = np.random.default_rng(seed)
    angles = rng.normal(0.6, 0.03, (4, 4000))
    stored = angles + 2 * np.pi * np.array([-1, 0, 1, 2])[:, None]
    return {
        "description": "Independent normal draws plus a fixed full-turn offset per chain; not MCMC.",
        "raw": diagnostics(stored),
        "wrapped": diagnostics(wrap(stored)),
        "sin": diagnostics(np.sin(stored)),
        "cos": diagnostics(np.cos(stored)),
    }


def plot_traces(traces, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    configurations = [
        ("intercept_aliases", "eta", 0, "Period-separated intercepts: raw predictor"),
        ("intercept_aliases", "wrapped", 0, "Same draws after wrapping"),
        ("at_branch_cut", "eta", 0, "Boundary-centered posterior: raw predictor"),
        ("at_branch_cut", "wrapped", 0, "Same draws wrapped at ±π"),
        (
            "slope_aliases",
            "wrapped",
            0,
            "Aliased slopes: predictions at observed x = 1",
        ),
        (
            "slope_aliases",
            "wrapped",
            1,
            "Same slopes: predictions at unobserved x = 0.5",
        ),
    ]
    for ax, (scenario, variable, probe, title) in zip(axes.flat, configurations):
        values = traces[scenario][variable][..., probe]
        for chain, draws in enumerate(values):
            ax.plot(draws[:250], lw=0.65, alpha=0.75, label=f"Chain {chain + 1}")
        ax.set(title=title, xlabel="Post-warmup draw", ylabel="Radians")
    axes[0, 0].legend(ncols=4, fontsize=8, loc="lower left")
    fig.suptitle(
        "Wrapping changes the representation; the samples here are identical",
        fontsize=15,
    )
    fig.savefig(output / "wrap_traces.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=750)
    parser.add_argument("--replicates", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("pymc", "pytensor", "arviz-stats", "numpy", "bambi")
        },
        "graph_equivalence": graph_equivalence(),
        "numerical_stress": numerical_stress(),
        "representation_control": representation_control(),
        "fits": [],
        "paired_draw_comparisons": [],
    }
    traces = {}
    for replicate in range(args.replicates):
        for scenario in SCENARIOS:
            for link in ("identity", "wrap"):
                print(f"Starting {scenario}, {link}, replicate {replicate}", flush=True)
                result, trace = fit_one(scenario, 2400 + replicate, link, args.draws, args.tune)
                results["fits"].append(result)
                if link == "identity":
                    identity_trace = trace
                    identity_result = result
                else:
                    results["paired_draw_comparisons"].append(
                        {
                            "scenario": scenario,
                            "seed": 2400 + replicate,
                            "max_abs_coefficient_draw_difference": float(
                                np.max(
                                    np.abs(trace["coefficients"] - identity_trace["coefficients"])
                                )
                            ),
                            "max_abs_angular_prediction_draw_difference": float(
                                np.max(np.abs(wrap(trace["eta"] - identity_trace["eta"])))
                            ),
                            "nuts_step_count_difference": result["total_nuts_steps"]
                            - identity_result["total_nuts_steps"],
                        }
                    )
                if replicate == 0 and link == "identity":
                    traces[scenario] = trace
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "link": link,
                            "divergences": result["divergences"],
                            "rhat_coefficients": result["coefficients"]["rhat"],
                            "rhat_reported": result["reported_mu"]["rhat"],
                            "rhat_wrapped": result["posthoc_wrapped_mu"]["rhat"],
                        }
                    ),
                    flush=True,
                )
    plot_traces(traces, args.output)


if __name__ == "__main__":
    main()
