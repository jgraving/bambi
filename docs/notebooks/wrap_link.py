"""Fit circular regression with mean directions reported in [-pi, pi).

Use the wrap link to give equivalent directions a common representation in
predictions and traces. The von Mises density is already periodic, so wrapping
preserves the likelihood and posterior with identical coefficient priors;
the paired experiments found no measurable sampling benefit in the tested models.

Inspect both sin(mu) and cos(mu) near the interval boundary and use circular
summaries. Regression coefficients stay on their original scale; identity
remains the default link. Angles are in radians.

Run with ``python docs/notebooks/wrap_link.py``. See
docs/validation/wrap/README.md for the experiments and practical guidance.
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
