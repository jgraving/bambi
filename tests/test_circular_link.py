"""Circular link values, likelihood gradients, and new-data predictions."""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

import bambi as bmb
from bambi.backend.pymc.links import wrap


@pytest.mark.parametrize("shape", [(), (7,), (1, 7)])
def test_wrap_values(shape):
    values = np.array([-3 * np.pi, -np.pi, -0.3, 0, 0.3, np.pi, 3 * np.pi])
    expected = np.array([-np.pi, -np.pi, -0.3, 0, 0.3, -np.pi, -np.pi])
    if not shape:
        values, expected = values[0], expected[0]
    else:
        values, expected = values.reshape(shape), expected.reshape(shape)
    link = bmb.Link("wrap")
    for result in (
        link.link(values),
        link.inverse_link(values),
        wrap(pt.as_tensor(values)).eval(),
    ):
        np.testing.assert_allclose(result, expected, atol=1e-14)
        assert np.shape(result) == shape


def test_wrap_periodicity_and_likelihood_gradient():
    eta = pt.dvector("eta")
    wrapped = wrap(eta)
    logp = pm.logp(pm.VonMises.dist(mu=wrapped, kappa=2), 0.4)
    evaluate = pytensor.function([eta], [wrapped, logp, pt.grad(logp.sum(), eta)])
    # Include points on both sides of the branch cut, but not the discontinuity itself.
    values = np.array([-np.pi - 1e-6, -0.3, -1e-20, 0, 1e-20, np.pi - 1e-6, np.pi + 1e-6])
    angles, density, gradient = evaluate(values)
    np.testing.assert_allclose(density, pm.logp(pm.VonMises.dist(mu=values, kappa=2), 0.4).eval())
    np.testing.assert_allclose(gradient, 2 * np.sin(0.4 - values))
    for turns in (-100, -1, 1, 100):
        np.testing.assert_allclose(evaluate(values + turns * 2 * np.pi)[0], angles, atol=1e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_wrap_preserves_principal_angles(dtype):
    pi = np.asarray(np.pi, dtype=dtype)
    values = np.array(
        [
            np.nextafter(-pi, 0, dtype=dtype),
            -1e-20,
            0,
            1e-20,
            np.nextafter(pi, 0, dtype=dtype),
        ],
        dtype=dtype,
    )
    link = bmb.Link("wrap")
    for result in (link.link(values), link.inverse_link(values), wrap(pt.as_tensor(values)).eval()):
        np.testing.assert_array_equal(result, values)


def test_wrap_half_open_boundary():
    values = np.array(
        [
            np.nextafter(-np.pi, -np.inf),
            -np.pi,
            np.nextafter(-np.pi, np.inf),
            np.nextafter(np.pi, -np.inf),
            np.pi,
            np.nextafter(np.pi, np.inf),
        ]
    )
    link = bmb.Link("wrap")
    for result in (link.link(values), link.inverse_link(values), wrap(pt.as_tensor(values)).eval()):
        assert np.all(result >= -np.pi)
        assert np.all(result < np.pi)
        np.testing.assert_allclose(np.exp(1j * result), np.exp(1j * values), atol=1e-15)
    eta = pt.dvector("eta")
    logp = pm.logp(pm.VonMises.dist(mu=wrap(eta), kappa=2), 0.4)
    gradient = pytensor.function([eta], pt.grad(logp.sum(), eta))(values)
    np.testing.assert_allclose(gradient, 2 * np.sin(0.4 - values), atol=1e-14)


@pytest.mark.parametrize("link", ["wrap", {"mu": "wrap", "kappa": "log"}])
def test_wrap_prediction(link):
    data = pd.DataFrame({"y": [0.2, -0.4, 0.5], "x": [-8.0, 0.0, 8.0]})
    model = bmb.Model("y ~ 0 + x", data, family="vonmises", link=link, priors={"kappa": 2})
    model.build()
    draws = xr.Dataset({"x": (("chain", "draw"), [[1.0, -2.0]])})
    idata = xr.DataTree.from_dict({"posterior": draws})
    for new_data in (None, pd.DataFrame({"x": [-12.0, 12.0], "y": [0.1, -0.2]})):
        frame = data if new_data is None else new_data
        eta = np.array([1.0, -2.0])[:, None] * frame.x.to_numpy()
        expected = np.angle(np.exp(1j * eta))
        prediction = model.predict(idata, data=new_data, inplace=False)
        group = prediction.posterior if new_data is None else prediction.predictions
        np.testing.assert_allclose(group.mu.values, expected[None])
        likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
        direct = pm.logp(pm.VonMises.dist(mu=eta, kappa=2), frame.y.to_numpy()).eval()
        np.testing.assert_allclose(likelihood.log_likelihood.y.values, direct[None])


def test_wrap_group_effects_with_concentration_regression():
    data = pd.DataFrame({"y": [0.2, -0.4, 0.5], "x": [-8.0, 0.0, 8.0], "group": ["a", "b", "a"]})
    model = bmb.Model(
        bmb.Formula("y ~ x + (1|group)", "kappa ~ 0 + x"),
        data,
        family="vonmises",
        link={"mu": "wrap", "kappa": "logexpm1"},
        center_predictors=False,
    )
    model.build()
    draws = xr.Dataset(
        {
            "Intercept": (("chain", "draw"), [[0.4, -0.2]]),
            "x": (("chain", "draw"), [[1.0, -2.0]]),
            "1|group": (("chain", "draw", "group_dim"), [[[0.6, -0.5], [0.1, 0.7]]]),
            "1|group_sigma": (("chain", "draw"), [[1.0, 1.0]]),
            "kappa_x": (("chain", "draw"), [[0.05, -0.1]]),
        },
        coords={"group_dim": ["a", "b"]},
    )
    idata = xr.DataTree.from_dict({"posterior": draws})
    new_data = pd.DataFrame({"x": [-12.0, 12.0], "group": ["b", "a"], "y": [0.1, -0.2]})
    for frame in (data, new_data):
        group_effect = draws["1|group"].sel(group_dim=frame.group.to_numpy()).values
        eta = (
            draws.Intercept.values[..., None]
            + draws.x.values[..., None] * frame.x.to_numpy()
            + group_effect
        )
        kappa = np.logaddexp(0, draws.kappa_x.values[..., None] * frame.x.to_numpy())
        prediction = model.predict(
            idata, data=frame, kind="response", inplace=False, random_seed=31
        )
        np.testing.assert_allclose(prediction.predictions.mu, np.angle(np.exp(1j * eta)))
        np.testing.assert_allclose(prediction.predictions.kappa, kappa)
        assert prediction.predictions.y.shape == eta.shape
        assert bool((np.abs(prediction.predictions.y) <= np.pi).all())
        likelihood = model.compute_log_likelihood(idata, data=frame, inplace=False)
        direct = pm.logp(pm.VonMises.dist(mu=eta, kappa=kappa), frame.y.to_numpy()).eval()
        np.testing.assert_allclose(likelihood.log_likelihood.y, direct)
    xr.testing.assert_equal(idata.posterior.to_dataset(), draws)


def test_circular_link_scope_and_default():
    data = pd.DataFrame({"y": [0.2, -0.4]})
    assert bmb.Model("y ~ 1", data, family="vonmises").family.link["mu"].name == "identity"
    for family, link in (
        ("gaussian", "wrap"),
        ("vonmises", {"kappa": "wrap"}),
    ):
        with pytest.raises(ValueError, match="cannot be used"):
            bmb.Model("y ~ 1", data, family=family, link=link)
