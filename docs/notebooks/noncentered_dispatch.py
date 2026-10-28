"""Register an external non-centered Student-t group-effect builder.

The built-in registry supports Normal priors with a sigma hyperprior.
Use a per-prior ``noncentered=False`` override for centered effects, such as
VonMises angular offsets whose inferred concentration remains in their prior.
This example does not claim a non-centered von Mises parametrization.

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
