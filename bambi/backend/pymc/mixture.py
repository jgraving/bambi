"""PyMC finite-mixture construction and aligned component views."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.utils import get_distribution_from_likelihood


def make_mixture_distribution(family):
    """Build a PyMC constructor that marginalizes the component index."""
    distribution = get_distribution_from_likelihood(family.component.likelihood)
    transform = transforms_registry.get_parameter_transform(family.component)

    def components(kwargs):
        parameters = {}
        for name in family.component.likelihood.params:
            value = pt.as_tensor_variable(kwargs.pop(name))
            if name not in family.component_parameters:
                value = value[..., None]
            parameters[name] = value
        if transform:
            parameters = transform(parameters)
        return distribution.dist(**parameters)

    class MixtureDistribution:
        def __new__(cls, name, weights, **kwargs):
            component_dist = components(kwargs)
            return pm.Mixture(name, w=weights, comp_dists=component_dist, **kwargs)

        @classmethod
        def dist(cls, weights, **kwargs):
            component_dist = components(kwargs)
            return pm.Mixture.dist(w=weights, comp_dists=component_dist, **kwargs)

    return MixtureDistribution


def build_ordered_views(family, parameters, model):
    """Apply one population-derived permutation to every component axis."""
    if family.order_by is None:
        return parameters
    if family.order_by not in model:
        raise ValueError(f"Unknown mixture ordering coefficient: {family.order_by!r}.")
    reference = model[family.order_by]
    if model.named_vars_to_dims.get(family.order_by) != ("__component__",):
        raise ValueError(
            "'order_by' must have exactly one value per component and no observation axis."
        )
    if family.component.likelihood.name == "VonMises":
        reference = (reference + np.pi) % (2 * np.pi) - np.pi
    with model:
        order = pm.Deterministic("component_order", pt.argsort(reference), dims="__component__")
        ordered = {}
        for variable in list(model.free_RVs) + list(model.deterministics):
            if variable is order:
                continue
            dims = model.named_vars_to_dims[variable.name]
            value = variable
            if "__component_reduced__" in dims:
                # Restore the fixed first logit before applying the common permutation.
                axis = dims.index("__component_reduced__")
                shape = tuple(1 if i == axis else value.shape[i] for i in range(value.ndim))
                value = pt.concatenate([pt.zeros(shape, dtype=value.dtype), value], axis=axis)
                dims = tuple("__component__" if d == "__component_reduced__" else d for d in dims)
            elif "__component__" not in dims:
                continue
            axis = dims.index("__component__")
            ordered[variable.name] = pm.Deterministic(
                variable.name + "_ordered", pt.take(value, order, axis=axis), dims=dims
            )
            if variable.name in model.__bambi_attrs__["offset_names"]:
                model.__bambi_attrs__["offset_names"].add(ordered[variable.name].name)
        return {name: ordered.get(name, value) for name, value in parameters.items()}


def mixture_weights(predictor, _parameters, inverse_link):
    """Normalize K logits with a fixed zero reference logit."""
    reference = pt.zeros((*predictor.shape[:-1], 1), dtype=predictor.dtype)
    return inverse_link(pt.concatenate([reference, predictor], axis=-1))
