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
    values = np.array([-np.pi - 1e-6, -0.3, np.pi - 1e-6, np.pi + 1e-6])
    angles, density, gradient = evaluate(values)
    np.testing.assert_allclose(density, pm.logp(pm.VonMises.dist(mu=values, kappa=2), 0.4).eval())
    np.testing.assert_allclose(gradient, 2 * np.sin(0.4 - values))
    for turns in (-100, -1, 1, 100):
        np.testing.assert_allclose(evaluate(values + turns * 2 * np.pi)[0], angles, atol=1e-12)


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


def test_circular_link_scope_and_default():
    data = pd.DataFrame({"y": [0.2, -0.4]})
    assert bmb.Model("y ~ 1", data, family="vonmises").family.link["mu"].name == "identity"
    for family, link in (
        ("gaussian", "wrap"),
        ("vonmises", {"kappa": "wrap"}),
    ):
        with pytest.raises(ValueError, match="cannot be used"):
            bmb.Model("y ~ 1", data, family=family, link=link)
