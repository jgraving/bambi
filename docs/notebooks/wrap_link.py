"""Circular regression with an explicitly wrapped mean direction.

Run from the repository with ``python docs/notebooks/wrap_link.py``.
Angles and coefficients are in radians. The wrap link maps the
predictor to [-pi, pi); identity remains the default for vonmises.

Wrapping puts equivalent directions in a common interval, which makes
directional traces and diagnostics comparable when chains differ by full
turns. The von Mises density is already periodic in its mean direction;
with identical coefficient priors the likelihood and posterior are unchanged.
Near the interval boundary, use circular summaries and inspect sin(mu) and
cos(mu), or angular deviations about a common reference direction. Regression
coefficients remain on their original scale. See docs/validation/wrap/README.md
for paired sampling experiments and the distinction between directions and slopes.
"""

import numpy as np
import pandas as pd

import bambi as bmb


def main():
    rng = np.random.default_rng(123)
    x = np.linspace(-2, 2, 100)
    data = pd.DataFrame({"x": x, "angle": rng.vonmises(1 + 2 * x, 4)})
    model = bmb.Model(
        "angle ~ x",
        data,
        family="vonmises",
        link="wrap",
        priors={
            "Intercept": bmb.Prior("Normal", mu=1, sigma=0.5),
            "x": bmb.Prior("Normal", mu=2, sigma=0.5),
        },
    )
    idata = model.fit(draws=500, tune=500, chains=2, random_seed=123)
    model.predict(idata, kind="response_params")
    assert bool(((idata.posterior.mu >= -np.pi) & (idata.posterior.mu < np.pi)).all())
    return model, idata


if __name__ == "__main__":
    main()
