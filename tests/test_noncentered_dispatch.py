"""Distribution dispatch and preservation of hierarchical prior semantics."""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest

import bambi as bmb
from bambi.backend.pymc.noncentered import location_scale_offsets, make_noncentered
from bambi.backend.pymc.terms.group_specific import build_distribution


@pytest.fixture
def build_prior():
    def build(prior, noncentered=True):
        model = pm.Model(coords={"group": range(3), "term": range(2)})
        model.__bambi_attrs__ = {"offset_names": set()}
        variable = build_distribution(
            prior, "effect", ("group",), ("term",), (), noncentered, {}, model
        )
        return model, variable

    return build


@pytest.mark.parametrize("mu", [2.5, np.array([1.0, -2.0])])
def test_normal_preserves_location_and_dimensions(build_prior, mu):
    model, variable = build_prior(
        bmb.Prior("Normal", mu=mu, sigma=bmb.Prior("HalfNormal", sigma=1))
    )
    evaluate = pytensor.function([model["effect_offset"], model["effect_sigma"]], variable)
    offsets = np.arange(6).reshape(3, 2).astype(float)
    scales = np.array([0.5, 2.0])
    np.testing.assert_allclose(evaluate(offsets, scales), mu + offsets * scales)
    assert model.named_vars_to_dims["effect"] == ("group", "term")
    assert model.named_vars_to_dims["effect_sigma"] == ("term",)
    assert model.__bambi_attrs__["offset_names"] == {"effect_offset"}


def test_random_location_is_preserved(build_prior):
    model, variable = build_prior(
        bmb.Prior(
            "Normal",
            mu=bmb.Prior("Normal", mu=1, sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=1),
        )
    )
    evaluate = pytensor.function(
        [model["effect_offset"], model["effect_mu"], model["effect_sigma"]], variable
    )
    np.testing.assert_allclose(evaluate(np.zeros((3, 2)), [1.5, -0.5], [1, 2]), [[1.5, -0.5]] * 3)


def test_custom_distribution_does_not_dispatch_by_prior_name(build_prior):
    def custom_normal(name, mu, sigma, dims):
        return pm.Normal(name, mu=mu, sigma=sigma, dims=dims)

    prior = bmb.Prior("Normal", dist=custom_normal, mu=0, sigma=bmb.Prior("HalfNormal", sigma=1))
    with pytest.raises(NotImplementedError, match="registered for custom_normal"):
        build_prior(prior)
    model, _ = build_prior(prior, noncentered=False)
    assert not model.__bambi_attrs__["offset_names"]


@pytest.fixture
def student_builder(monkeypatch):
    # Restore the registry after testing an extension, including the unregistered fallback.
    monkeypatch.setattr(make_noncentered, "builders", make_noncentered.builders.copy())

    @make_noncentered.register(pm.StudentT, recover=location_scale_offsets)
    def builder(label, dims, model, nu, mu, sigma):
        with model:
            offset = pm.StudentT(label + "_offset", nu=nu, mu=0, sigma=1, dims=dims)
            model.__bambi_attrs__["offset_names"].add(offset.name)
            return pm.Deterministic(label, mu + sigma * offset, dims=dims)

    return builder


def test_external_builder(build_prior, student_builder):
    model, variable = build_prior(
        bmb.Prior("StudentT", nu=4, mu=2, sigma=bmb.Prior("HalfNormal", sigma=1))
    )
    evaluate = pytensor.function([model["effect_offset"], model["effect_sigma"]], variable)
    np.testing.assert_allclose(evaluate(np.ones((3, 2)), [0.5, 2.0]), [[2.5, 4.0]] * 3)
    assert make_noncentered.builders[pm.StudentT][0] is student_builder


@pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
def test_external_builder_prediction_and_offset_omission(
    student_builder, mock_pymc_sample, distribution
):
    data = pd.DataFrame({"y": [0.1, 0.2, -0.3, 0.4], "g": ["a", "a", "b", "b"]})
    kwargs = {"nu": 4} if distribution == "StudentT" else {}
    prior = bmb.Prior(distribution, mu=1, sigma=bmb.Prior("HalfNormal", sigma=1), **kwargs)
    model = bmb.Model("y ~ 0 + (1|g)", data, priors={"1|g": prior})
    idata = model.fit(draws=10, chains=1, omit_offsets=True)
    assert "1|g_offset" not in idata.posterior
    prediction = model.predict(idata, inplace=False)
    np.testing.assert_allclose(
        prediction.posterior.mu.values[..., :2],
        np.repeat(idata.posterior["1|g"].values[..., :1], 2, axis=-1),
    )
    for new_data in (None, data.iloc[[0, 2, 3]]):
        frame = data if new_data is None else new_data
        indices = np.array([0 if group == "a" else 1 for group in frame.g])
        expected_mu = idata.posterior["1|g"].values[..., indices]
        predicted = model.predict(idata, data=new_data, inplace=False)
        group = predicted.posterior if new_data is None else predicted.predictions
        np.testing.assert_allclose(group.mu.values, expected_mu)
        likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
        expected = pm.logp(
            pm.Normal.dist(mu=expected_mu, sigma=idata.posterior.sigma.values[..., None]),
            frame.y.to_numpy(),
        ).eval()
        np.testing.assert_allclose(likelihood.log_likelihood.y.values, expected)


def test_centered_circular_and_noncentered_normal_effects():
    data = pd.DataFrame({"y": [0.1, 0.2, -0.3, 0.4], "g": ["a", "a", "b", "b"]})
    prior = bmb.Prior("VonMises", mu=0, kappa=bmb.Prior("HalfNormal", sigma=1), noncentered=False)
    model = bmb.Model(
        bmb.Formula("y ~ 1 + (1|g)", "kappa ~ 1 + (1|g)"),
        data,
        family="vonmises",
        priors={"1|g": prior},
    )
    model.build()
    assert model.backend.model.__bambi_attrs__["offset_names"] == {"kappa_1|g_offset"}
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))


def test_unsupported_normal_parametrization_is_explicit(build_prior):
    with pytest.raises(NotImplementedError, match="require a sigma hyperprior"):
        build_prior(bmb.Prior("Normal", mu=bmb.Prior("Normal", mu=0, sigma=1), sigma=1))
