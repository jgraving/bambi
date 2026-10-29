"""Register an external non-centered Student-t group-effect builder.

The built-in registry supports Normal priors with a sigma hyperprior.
Use a per-prior ``noncentered=False`` override for centered effects, such as
VonMises angular offsets whose inferred concentration remains in their prior.
This example does not claim a non-centered von Mises parametrization.

The location and scale are retained exactly: ``effect = mu + sigma * offset``.
Registering a builder also requires an inverse so Bambi can recover omitted
offset draws for prediction and log likelihood. For this location-scale
transform, ``location_scale_offsets`` computes ``(effect - mu) / sigma`` using
the posterior hyperparameter draws. Nonlinear transformations need their own
inverse. For example, a LogNormal builder using ``exp(mu + sigma * offset)``
would recover ``(log(effect) - mu) / sigma`` instead.

The inverse selected when a model is built stays attached to that model, even
if the registry is later changed. Auxiliary variable names must be registered
in ``offset_names``; they need not end in ``_offset``. Predictions for new group
levels draw from the original distribution's ``.dist`` method conditional on
the fitted hyperparameters, so a custom prior callable must provide that
method to support unseen groups.

Run with ``python docs/notebooks/noncentered_dispatch.py``.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

import bambi as bmb
from bambi.backend.pymc.noncentered import location_scale_offsets, make_noncentered


@make_noncentered.register(pm.StudentT, recover=location_scale_offsets)
def student_noncentered(label, dims, model, nu, mu, sigma):
    """Build location-scale Student-t effects with standardized offsets.

    This extension supports a fixed number of degrees of freedom and an
    inferred scale. Other parametrizations must supply a different builder.
    """
    if isinstance(nu, pt.TensorVariable) or not isinstance(sigma, pt.TensorVariable):
        raise NotImplementedError("Expected fixed nu and a sigma hyperprior.")
    with model:
        offset = pm.StudentT(label + "_offset", nu=nu, mu=0, sigma=1, dims=dims)
        model.__bambi_attrs__["offset_names"].add(offset.name)
        return pm.Deterministic(label, mu + sigma * offset, dims=dims)


def main():
    rng = np.random.default_rng(123)
    data = pd.DataFrame({"y": rng.normal(size=40), "group": np.repeat(list("abcd"), 10)})
    model = bmb.Model(
        "y ~ 1 + (1|group)",
        data,
        priors={
            "1|group": bmb.Prior("StudentT", nu=4, mu=0, sigma=bmb.Prior("HalfNormal", sigma=1))
        },
    )
    model.build()
    return model


if __name__ == "__main__":
    main()
