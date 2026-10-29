"""Finite mixtures with a common scalar component family."""

import ast
from copy import deepcopy

import numpy as np

from bambi.families.family import Family
from bambi.families.likelihood import Likelihood
from bambi.families.types import DimType, ParamSpec, ResponseType
from bambi.priors import Prior


class Mixture(Family):
    """A finite mixture of copies of a scalar family.

    Parameters
    ----------
    component : str or Family
        Component family. Its response must be numeric and scalar, and its
        parameters must be scalar before introducing the mixture axis.
        Custom distributions must provide a PyMC-compatible ``.dist``
        constructor. Registered response-data transforms are not supported.
    n : int
        Number of components, at least two.
    order_by : str or None, optional
        Name of a population coefficient with one value per component, such
        as ``"Intercept"``. Its order defines additional ``*_ordered`` views
        of all component parameters and coefficients, including group effects.
        This is a reporting convention, not an identifying prior.
        Use the coefficient's alias if one is assigned with ``Model.set_alias``.

    Notes
    -----
    Use ``Formula("y ~ mixture(mu)", "mu ~ x", "weights ~ w")``.
    Parameters listed in ``mixture(...)`` have distinct coefficients per
    component; other parameters are shared. Weight regressions use softmax
    with the first logit fixed at zero. Without a weights formula, weights
    have a uniform Dirichlet prior.

    This initial API supports unindexed formulas. Indexed component formulas,
    heterogeneous families, and automatic identifying priors are not provided.
    Component parameters and weights are available through ``Model.predict``;
    weights are prior class probabilities, not posterior responsibilities
    conditional on the observed response.

    Examples
    --------
    >>> family = Mixture("vonmises", n=2, order_by="Intercept")
    >>> family.n
    2
    """

    def __init__(self, component, n, order_by=None):
        # pylint: disable=import-outside-toplevel
        from bambi.defaults.families import get_builtin_family

        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError("'n' must be an integer greater than or equal to two.")
        component = get_builtin_family(component) if isinstance(component, str) else component
        if not isinstance(component, Family) or isinstance(component, Mixture):
            raise TypeError("'component' must be a non-mixture Family or a built-in family name.")
        if component.DATA_TYPE != ResponseType.NUMERIC or component.RESPONSE_NDIM != 0:
            raise ValueError("Mixtures currently require a numeric scalar response family.")
        if any(component.get_param_spec(p).ndim for p in component.likelihood.params):
            raise ValueError("Mixtures currently require scalar component parameters.")
        if "weights" in component.likelihood.params:
            raise ValueError("'weights' is reserved for mixture probabilities.")
        if order_by is not None and not isinstance(order_by, str):
            raise TypeError("'order_by' must be a coefficient name or None.")
        self.component = deepcopy(component)
        self.n = n
        self.order_by = order_by
        self.component_parameters = ()
        self.PARAMETERS = {  # pylint: disable=invalid-name
            p: self.component.get_param_spec(p) for p in self.component.likelihood.params
        } | {"weights": ParamSpec(["softmax"], ndim=1, coefs_dim=DimType.MIXTURE_REDUCED)}
        likelihood = Likelihood(
            f"Mixture{component.likelihood.name}",
            params=(*component.likelihood.params, "weights"),
            parent=component.likelihood.parent,
        )
        super().__init__(
            f"mixture_{component.name}", likelihood, component.link | {"weights": "softmax"}
        )
        self.default_priors = deepcopy(component.default_priors)
        self.default_priors["weights"] = Prior("Dirichlet", a=np.ones(n))


class MixtureVonMises(Mixture):
    """A finite von Mises mixture in radians.

    Parameters
    ----------
    n : int
        Number of directional components, at least two.
    order_by : str or None, optional
        Population coefficient used to label components. The ``mu`` intercept
        is wrapped to ``[-pi, pi)`` before computing the order; slopes and
        other coefficients retain their numerical order. The same indices
        are applied to every component parameter and group-specific effect.
        Use the coefficient's alias when one is assigned. The branch cut
        remains: these views can jump when a direction crosses the boundary
        and do not establish circular identifiability.

    Examples
    --------
    >>> family = MixtureVonMises(n=2, order_by="Intercept")
    >>> family.name
    'mixture_vonmises'
    """

    def __init__(self, n, order_by=None):
        super().__init__("vonmises", n, order_by)


def prepare_mixture_formula(family, formula):
    """Configure a model-local mixture and normalize its component formulas."""
    from bambi.formula import Formula  # pylint: disable=import-outside-toplevel

    if not isinstance(family, Mixture):
        return family, formula
    if formula.is_nonlinear:
        raise NotImplementedError("Nonlinear mixture formulas are not supported yet.")
    response, _, rhs = formula.main.partition("~")
    try:
        call = ast.parse(rhs.strip(), mode="eval").body
    except SyntaxError as error:
        raise ValueError("Use 'y ~ mixture(parameter, ...)' for a mixture family.") from error
    if (  # pylint: disable=too-many-boolean-expressions
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "mixture"
        or not call.args
        or call.keywords
        or any(not isinstance(arg, ast.Name) for arg in call.args)
    ):
        raise ValueError("Use 'y ~ mixture(parameter, ...)' with plain parameter names.")
    names = tuple(arg.id for arg in call.args)
    if len(names) != len(set(names)) or set(names) - set(family.component.likelihood.params):
        raise ValueError("mixture(...) must contain distinct component-family parameter names.")
    if len(set(formula.additionals_lhs)) != len(formula.additionals_lhs):
        raise ValueError("Duplicate mixture parameter formulas are not allowed.")
    family = deepcopy(family)
    family.component_parameters = names
    for name in names:
        family.PARAMETERS[name] = family.PARAMETERS[name]._replace(
            ndim=1, coefs_dim=DimType.MIXTURE
        )
    formulas = dict(zip(formula.additionals_lhs, formula.additionals, strict=True))
    parent = family.likelihood.parent
    parent_formula = formulas.pop(parent, f"{parent} ~ 1").partition("~")[2]
    # Every component-varying parameter has a coefficient formula by default.
    for name in names:
        if name != parent:
            formulas.setdefault(name, f"{name} ~ 1")
    return family, Formula(f"{response.strip()} ~ {parent_formula.strip()}", *formulas.values())
