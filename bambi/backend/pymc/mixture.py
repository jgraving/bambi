"""PyMC finite-mixture construction and aligned component views."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.utils import get_distribution_from_likelihood


def make_mixture_distribution(family):
    """Build a PyMC constructor that marginalizes the component index."""
    if transforms_registry.get_data_transform(family.component) is not None:
        raise NotImplementedError(
            "Mixtures of families with response-data transforms are not supported yet."
        )
    distribution = get_distribution_from_likelihood(family.component.likelihood)
    if not callable(getattr(distribution, "dist", None)):
        raise TypeError("A mixture component distribution must provide a '.dist' constructor.")
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


def build_ordered_views(family, parameters, model, conditional_parameters):
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
        location = conditional_parameters.get("mu")
        if location is not None and location.intercept_term is not None:
            intercept = location.intercept_term.label
            if family.order_by in (intercept, intercept + "_centered"):
                wrapped = (reference + np.pi) % (2 * np.pi) - np.pi
                wrapped = pt.switch(wrapped >= np.pi, wrapped - 2 * np.pi, wrapped)
                # Preserve tiny differences within the canonical interval;
                # adding pi can otherwise round distinct directions to a tie.
                reference = pt.switch(
                    (reference >= -np.pi) & (reference < np.pi), reference, wrapped
                )
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
            ordered[variable] = pm.Deterministic(
                variable.name + "_ordered", pt.take(value, order, axis=axis), dims=dims
            )
            if variable.name in model.__bambi_attrs__["offset_names"]:
                model.__bambi_attrs__["offset_names"].add(ordered[variable].name)
        # Parameter keys retain their formula names, whereas graph variables may
        # have aliases. Match the variables themselves so a rename cannot cause
        # only some parameters of a component to be permuted in the likelihood.
        return {name: ordered.get(value, value) for name, value in parameters.items()}


def mixture_weights(predictor, _parameters, inverse_link):
    """Normalize K logits with a fixed zero reference logit."""
    reference = pt.zeros((*predictor.shape[:-1], 1), dtype=predictor.dtype)
    return inverse_link(pt.concatenate([reference, predictor], axis=-1))
