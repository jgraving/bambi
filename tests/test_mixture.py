"""Finite-mixture densities, dimensions, predictions, and aligned reporting."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import bambi as bmb


@pytest.fixture
def circular_data():
    return pd.DataFrame(
        {"y": [-0.8, 0.3, 0.1, 1.2], "x": [-1.0, 0.0, 1.0, 2.0], "animal": ["a", "a", "b", "b"]}
    )


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("varying", ["mu", "kappa", "mu, kappa"])
@pytest.mark.parametrize("weights_formula", [False, True])
def test_mixture_likelihood_and_prediction(
    circular_data, mock_pymc_sample, n, varying, weights_formula
):
    formulas = ["mu ~ x", "kappa ~ x"]
    if weights_formula:
        formulas.append("weights ~ x")
    model = bmb.Model(
        bmb.Formula(f"y ~ mixture({varying})", *formulas),
        circular_data,
        family=bmb.MixtureVonMises(n),
    )
    idata = model.fit(draws=5, chains=1, random_seed=12)
    for data in (None, circular_data.iloc[:2]):
        result = model.predict(idata, data=data, inplace=False)
        draws = result.posterior if data is None else result.predictions
        # A marginal weights prior stays in posterior for new observations.
        weights = (draws if weights_formula else idata.posterior)["weights"].values
        mu, kappa = draws.mu.values, draws.kappa.values
        if "mu" not in varying:
            mu = mu[..., None]
        if "kappa" not in varying:
            kappa = kappa[..., None]
        if not weights_formula:
            weights = weights[..., None, :]
        np.testing.assert_allclose(weights.sum(axis=-1), 1)
        response = circular_data.y.to_numpy() if data is None else data.y.to_numpy()
        expected = pm.logp(
            pm.Mixture.dist(w=weights, comp_dists=pm.VonMises.dist(mu=mu, kappa=kappa)), response
        ).eval()
        actual = model.compute_log_likelihood(
            idata, data=data, inplace=False
        ).log_likelihood.y.values
        np.testing.assert_allclose(actual, expected)
        predictive = model.predict(
            idata, data=data, kind="response", inplace=False, random_seed=123
        )
        predicted = predictive.posterior_predictive if data is None else predictive.predictions
        assert predicted.y.shape == (1, 5, len(response))
        assert np.all(np.isfinite(predicted.y))


@pytest.mark.parametrize("sparse", [False, True])
def test_hierarchical_ordering_keeps_component_parameters_together(
    circular_data, mock_pymc_sample, monkeypatch, sparse
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse)
    circular = bmb.Prior(
        "VonMises", mu=0, kappa=bmb.Prior("HalfNormal", sigma=2), noncentered=False
    )
    model = bmb.Model(
        bmb.Formula(
            "y ~ mixture(mu, kappa)",
            "mu ~ 1 + (1|animal)",
            "kappa ~ 1 + (1|animal)",
            "weights ~ 1 + (1|animal)",
        ),
        circular_data,
        family=bmb.MixtureVonMises(2, order_by="Intercept"),
        priors={"1|animal": circular},
    )
    idata = model.fit(draws=10, chains=1, random_seed=41)
    assert not (
        model.backend.model.__bambi_attrs__["offset_names"] & set(idata.posterior.data_vars)
    )
    draws = model.predict(idata, inplace=False).posterior
    order = np.argsort((draws.Intercept.values + np.pi) % (2 * np.pi) - np.pi, axis=-1)
    for name in (
        "Intercept",
        "1|animal",
        "1|animal_kappa",
        "kappa_1|animal",
        "kappa_1|animal_sigma",
        "mu",
        "kappa",
        "weights",
    ):
        raw = draws[name].values
        indices = order.reshape((*order.shape[:2], *((1,) * (raw.ndim - 3)), 2))
        expected = np.take_along_axis(raw, np.broadcast_to(indices, raw.shape), axis=-1)
        np.testing.assert_allclose(draws[name + "_ordered"].values, expected)
    # Weight effects are represented as full logits including the fixed reference.
    raw = draws["weights_1|animal"].values
    full = np.concatenate([np.zeros_like(raw), raw], axis=-1)
    expected = np.take_along_axis(full, np.broadcast_to(order[:, :, None], full.shape), axis=-1)
    np.testing.assert_allclose(draws["weights_1|animal_ordered"].values, expected)
    for data in (circular_data.iloc[:2], circular_data.iloc[:2].assign(animal="new")):
        prediction = model.predict(idata, data=data, inplace=False, random_seed=42)
        assert prediction.predictions.mu_ordered.shape == (1, 10, 2, 2)


def test_ordering_branch_cut_and_density_invariance(circular_data):
    import pytensor

    model = bmb.Model(
        bmb.Formula("y ~ mixture(mu,kappa)", "mu ~ 1", "kappa ~ 1", "weights ~ 1"),
        circular_data,
        family=bmb.MixtureVonMises(2, order_by="Intercept"),
    )
    model.build()
    backend = model.backend.model
    evaluate = pytensor.function(
        [backend["Intercept"], backend["kappa_Intercept"], backend["weights_Intercept"]],
        [
            backend["component_order"],
            backend["mu_ordered"],
            backend["kappa_ordered"],
            backend["weights_ordered"],
        ],
    )
    orders = []
    for direction in (np.pi - 0.01, np.pi + 0.01):
        mu = np.array([direction, 0.0])
        order, sorted_mu, sorted_kappa, sorted_weights = evaluate(mu, [0.0, np.log(3)], [np.log(2)])
        orders.append(order)
        original = pm.Mixture.dist(
            w=[1 / 3, 2 / 3], comp_dists=pm.VonMises.dist(mu=mu, kappa=[1, 3])
        )
        sorted_dist = pm.Mixture.dist(
            w=sorted_weights, comp_dists=pm.VonMises.dist(mu=sorted_mu, kappa=sorted_kappa)
        )
        np.testing.assert_allclose(
            pm.logp(original, circular_data.y).eval(), pm.logp(sorted_dist, circular_data.y).eval()
        )
    np.testing.assert_array_equal(orders, [[1, 0], [0, 1]])


@pytest.mark.parametrize("component", ["gaussian", "beta"])
def test_generic_scalar_component_and_parameter_transform(
    circular_data, mock_pymc_sample, component
):
    model = bmb.Model(
        bmb.Formula("y ~ mixture(mu)", "mu ~ x", "weights ~ 1"),
        circular_data.assign(y=[0.2, 0.3, 0.4, 0.5]),
        family=bmb.Mixture(component, 2),
    )
    idata = model.fit(draws=5, chains=1, random_seed=3)
    assert np.isfinite(model.compute_log_likelihood(idata, inplace=False).log_likelihood.y).all()


def test_family_configuration_is_model_local(circular_data):
    family = bmb.MixtureVonMises(2)
    first = bmb.Model(bmb.Formula("y ~ mixture(mu)"), circular_data, family=family)
    second = bmb.Model(bmb.Formula("y ~ mixture(kappa)"), circular_data, family=family)
    assert family.component_parameters == ()
    assert first.family.component_parameters == ("mu",)
    assert second.family.component_parameters == ("kappa",)


@pytest.mark.parametrize("n", [True, 0, 1, 2.5])
def test_invalid_component_counts(n):
    with pytest.raises(ValueError, match="integer"):
        bmb.MixtureVonMises(n)


@pytest.mark.parametrize(
    "formula",
    ["y ~ x", "y ~ mixture()", "y ~ mixture(mu,mu)", "y ~ mixture(weights)", "y ~ mixture(mu) + x"],
)
def test_invalid_mixture_formula(circular_data, formula):
    with pytest.raises(ValueError, match="mixture"):
        bmb.Model(formula, circular_data, family=bmb.MixtureVonMises(2))


def test_invalid_ordering_reference(circular_data):
    model = bmb.Model(
        bmb.Formula("y ~ mixture(mu)"), circular_data, family=bmb.MixtureVonMises(2, order_by="mu")
    )
    with pytest.raises(ValueError, match="no observation axis"):
        model.build()
