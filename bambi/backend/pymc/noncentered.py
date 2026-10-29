"""Extensible construction of non-centered group-specific priors."""

import pymc as pm
import pytensor.tensor as pt
import xarray as xr


class NoncenteredRegistry:
    """Dispatch non-centered builders by the resolved distribution callable.

    Registration uses exact callable identity. A custom ``Prior.dist`` never
    inherits a builder merely because its prior is named ``"Normal"``.
    """

    def __init__(self):
        self.builders = {}

    def register(self, distribution, *, recover):
        """Register a builder for a PyMC distribution or custom prior callable.

        Parameters
        ----------
        distribution : callable
            The exact callable used to construct the centered prior.
        recover : callable
            ``recover(label, value, **kwargs)`` returns a mapping from offset
            names to their reconstructed draws. ``value`` and hyperparameters
            are xarray arrays with named dimensions. This inverse is required
            for predictions and log likelihood when ``omit_offsets=True``.
            It must invert the builder for every omitted auxiliary variable;
            a nonlinear transform generally needs its own inverse rather than
            ``location_scale_offsets``.

        Returns
        -------
        callable
            A decorator for a function with signature
            ``builder(label, dims, model, **kwargs)``. The function must validate
            its supported parametrization, create and return the coefficient
            variable, and add auxiliary variable names to
            ``model.__bambi_attrs__["offset_names"]`` for ``omit_offsets``.
            ``kwargs`` contains broadcast numeric arguments and built hyperpriors.

        Notes
        -----
        Registration replaces any previous builder for the same callable.
        Models keep the inverse selected when they were built, so later
        registrations do not change how their saved draws are reconstructed.
        Predicting newly observed group levels uses the original centered
        distribution's ``.dist`` method, not the non-centered builder.
        See ``docs/notebooks/noncentered_dispatch.py`` for an extension example.
        """
        if not callable(distribution):
            raise TypeError("A non-centered builder must be registered for a callable.")
        if not callable(recover):
            raise TypeError("A non-centered inverse must be callable.")

        def decorator(builder):
            if not callable(builder):
                raise TypeError("A non-centered builder must be callable.")
            self.builders[distribution] = (builder, recover)
            return builder

        return decorator

    def __call__(self, distribution, label, dims, model, **kwargs):
        entry = self.builders.get(distribution)
        if entry is None:
            name = getattr(distribution, "__name__", type(distribution).__name__)
            raise NotImplementedError(
                f"No non-centered parametrization is registered for {name}. "
                "Use Prior(..., noncentered=False) or register a builder with "
                "bambi.backend.pymc.noncentered.make_noncentered.register."
            )
        builder, recover = entry
        offsets_before = model.__bambi_attrs__["offset_names"].copy()
        variable = builder(label, dims, model, **kwargs)
        offsets = model.__bambi_attrs__["offset_names"] - offsets_before
        model.__bambi_attrs__.setdefault("noncentered", {})[label] = (
            recover,
            dims,
            kwargs,
            offsets,
        )
        return variable


make_noncentered = NoncenteredRegistry()


def location_scale_offsets(label, value, mu=0, sigma=1, **_kwargs):
    """Recover a standardized ``label + '_offset'`` from location-scale draws."""
    return {label + "_offset": (value - mu) / sigma}


def recover_offsets(posterior, model):
    """Reconstruct omitted auxiliary draws using each registered inverse."""
    recovered = {}
    for label, (recover, dims, kwargs, offsets) in model.__bambi_attrs__.get(
        "noncentered", {}
    ).items():
        missing = offsets - set(posterior.data_vars)
        if not missing:
            continue
        parameters = {
            name: (
                posterior[value.name]
                if isinstance(value, pt.TensorVariable)
                else xr.DataArray(value, dims=dims)
            )
            for name, value in kwargs.items()
        }
        values = recover(label, posterior[label], **parameters)
        unresolved = missing - values.keys()
        if unresolved:
            raise ValueError(
                f"The non-centered inverse for '{label}' did not recover {unresolved}."
            )
        recovered.update({name: values[name] for name in missing})
    return recovered


@make_noncentered.register(pm.Normal, recover=location_scale_offsets)
def normal_noncentered(label, dims, model, mu=0, sigma=None, **kwargs):
    """Construct ``mu + sigma * z`` with a standard Normal offset ``z``."""
    if not isinstance(sigma, pt.TensorVariable) or kwargs:
        raise NotImplementedError(
            "Non-centered Normal priors require a sigma hyperprior and accept only mu and sigma. "
            "Use Prior(..., noncentered=False) for other parametrizations."
        )
    with model:
        offset = pm.Normal(label + "_offset", mu=0, sigma=1, dims=dims)
        model.__bambi_attrs__["offset_names"].add(label + "_offset")
        return pm.Deterministic(label, mu + offset * sigma, dims=dims)
