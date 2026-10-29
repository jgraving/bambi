from sys import float_info

import numpy as np
import pytensor.tensor as pt


def wrap(x):
    """Wrap an angular predictor in radians to ``[-pi, pi)``."""
    result = (x + np.pi) % (2 * np.pi) - np.pi
    result = pt.switch(pt.ge(result, np.pi), result - 2 * np.pi, result)
    return pt.switch(pt.and_(pt.ge(x, -np.pi), pt.lt(x, np.pi)), x, result)


def probit(x):
    """Probit function that ensures result is in (0, 1)."""
    eps = float_info.epsilon
    result = 0.5 + 0.5 * pt.erf(x / pt.sqrt(2))
    result = pt.switch(pt.eq(result, 0), eps, result)
    result = pt.switch(pt.eq(result, 1), 1 - eps, result)

    return result


def cloglog(x):
    """Cloglog function that ensures result is in (0, 1)."""
    eps = float_info.epsilon
    result = 1 - pt.exp(-pt.exp(x))
    result = pt.switch(pt.eq(result, 0), eps, result)
    result = pt.switch(pt.eq(result, 1), 1 - eps, result)

    return result


def logit(x):
    """Logit function that ensures result is in (0, 1)."""
    eps = float_info.epsilon
    result = pt.sigmoid(x)
    result = pt.switch(pt.eq(result, 0), eps, result)
    result = pt.switch(pt.eq(result, 1), 1 - eps, result)
    return result


def identity(x):
    return x


def inverse_squared(x):
    return pt.reciprocal(pt.sqrt(x))
