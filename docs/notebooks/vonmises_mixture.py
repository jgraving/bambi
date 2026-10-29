"""A two-component circular mixture with animal-specific deviations.

The formula follows the finite-mixture proposal in bambinos/bambi#1040 and
#1041. The hierarchy is inspired by the public locust example in
https://github.com/jgraving/unwrap/blob/main/notebooks/locust_direction_mixture_model.ipynb.
This example uses synthetic data and does not reproduce that analysis.

The angular effects are centered von Mises effects with inferred concentration.
Normal effects on concentration and weight predictors remain non-centered.
The existing identity link is periodic inside the von Mises likelihood.

``order_by`` creates aligned ``*_ordered`` views from ONE population ordering.
It does not identify a circular mixture. Crossing the branch cut can change
the reported labels, including for two components. Inspect predictive density
and use circular summaries; sorted traces alone do not establish convergence.
Raw draws are retained. Never replace raw coefficients with ordered views when
predicting: the raw graph includes the reference-logit parametrization.
"""

import numpy as np
import pandas as pd

import bambi as bmb


def main():
    rng = np.random.default_rng(123)
    animals = np.repeat(np.arange(8), 30)
    components = rng.binomial(1, 0.6, len(animals))
    means = np.array([-1.0, 1.0])
    deviations = rng.vonmises(0, 30, size=(8, 2))
    data = pd.DataFrame(
        {
            "direction": rng.vonmises(means[components] + deviations[animals, components], 8),
            "animal": animals.astype(str),
        }
    )
    formula = bmb.Formula(
        "direction ~ mixture(mu, kappa)",
        "mu ~ 1 + (1|animal)",
        "kappa ~ 1 + (1|animal)",
        "weights ~ 1 + (1|animal)",
    )
    priors = {
        "Intercept": bmb.Prior("VonMises", mu=0, kappa=1e-6),
        "1|animal": bmb.Prior(
            "VonMises", mu=0, kappa=bmb.Prior("HalfNormal", sigma=30), noncentered=False
        ),
        "kappa": {"Intercept": bmb.Prior("Normal", mu=np.log(8), sigma=0.5)},
    }
    model = bmb.Model(
        formula, data, family=bmb.MixtureVonMises(2, order_by="Intercept"), priors=priors
    )
    model.build()
    # Supply separated initial directions when fitting this exchangeable mixture:
    # idata = model.fit(initvals={"Intercept": [-1, 1]}, target_accept=0.95)
    # model.predict(idata, kind="response_params")
    # idata.posterior["mu_ordered"] and ["weights_ordered"] use matching labels.
    return model


if __name__ == "__main__":
    main()
