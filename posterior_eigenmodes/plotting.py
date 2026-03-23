from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .results import CovarianceComparison
from .stats import select_sample_columns, weighted_mean


def get_mode_samples(
    reference_samples: object,
    alternative_samples: object,
    comparison: CovarianceComparison,
    *,
    parameter_names_a: Sequence[str] | None = None,
    parameter_names_b: Sequence[str] | None = None,
    center: bool = True,
    use_normalized_modes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all generalized mode coefficients for the two datasets.

    The returned arrays have shape ``(n_samples, n_modes)`` and can be plotted
    however the user prefers in the notebook.
    """

    reference_projection = project_samples_onto_modes(
        reference_samples,
        comparison,
        dataset="reference",
        parameter_names=parameter_names_a,
        center=center,
        use_normalized_modes=use_normalized_modes,
    )
    alternative_projection = project_samples_onto_modes(
        alternative_samples,
        comparison,
        dataset="alternative",
        parameter_names=parameter_names_b,
        center=center,
        use_normalized_modes=use_normalized_modes,
    )

    return reference_projection, alternative_projection


def project_samples_onto_modes(
    samples: object,
    comparison: CovarianceComparison,
    *,
    dataset: str = "reference",
    parameter_names: Sequence[str] | None = None,
    center: bool = True,
    use_normalized_modes: bool = True,
) -> np.ndarray:
    """Project samples onto the generalized comparison modes.

    The returned array has shape ``(n_samples, n_modes)``. When ``center=True``,
    each dataset is centered on its own weighted mean before projection.
    Using the unnormalized comparison modes means the projected variances are
    directly comparable to the generalized degradation factors.
    """

    matrix, names, weights = _coerce_samples(samples, parameter_names=parameter_names)

    if dataset == "reference":
        requested_names = comparison.parameter_names
    elif dataset == "alternative":
        requested_names = comparison.alternative_parameter_names
    else:
        raise ValueError("dataset must be either 'reference' or 'alternative'.")

    selected, _ = select_sample_columns(matrix, names, requested_names)
    if center:
        selected = selected - weighted_mean(selected, weights)

    basis = (
        comparison.normalized_mode_coefficients
        if use_normalized_modes
        else comparison.mode_vectors
    )
    projection = np.einsum("ni,ij->nj", selected, basis, optimize=True)

    if not np.all(np.isfinite(projection)):
        mode_type = "normalized" if use_normalized_modes else "raw"
        raise ValueError(
            f"Projection onto {mode_type} generalized modes produced non-finite values."
        )

    return projection


def plot_mode_distributions(
    reference_samples: object,
    alternative_samples: object,
    comparison: CovarianceComparison,
    *,
    parameter_names_a: Sequence[str] | None = None,
    parameter_names_b: Sequence[str] | None = None,
    mode_indices: Sequence[int] = (1,),
    labels: tuple[str, str] = ("reference", "alternative"),
    bins: int = 80,
):
    """Plot 1D histograms of selected generalized mode coefficients."""

    import matplotlib.pyplot as plt

    reference_projection, alternative_projection = get_mode_samples(
        reference_samples,
        alternative_samples,
        comparison,
        parameter_names_a=parameter_names_a,
        parameter_names_b=parameter_names_b,
        use_normalized_modes=True,
    )

    indices = tuple(mode_indices)
    if not indices:
        raise ValueError("mode_indices must contain at least one mode index.")

    fig, axes = plt.subplots(len(indices), 1, figsize=(7, 3 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for ax, mode_index in zip(axes, indices, strict=True):
        if mode_index < 1 or mode_index > comparison.mode_vectors.shape[1]:
            raise ValueError(f"Mode index {mode_index} is out of range.")

        idx = mode_index - 1
        ax.hist(
            reference_projection[:, idx],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=labels[0],
        )
        ax.hist(
            alternative_projection[:, idx],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=labels[1],
        )
        ax.set_title(
            f"Mode {mode_index}: rho = {comparison.degradation_factors[idx]:.3g}"
        )
        ax.set_xlabel("Mode coefficient")
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    return fig, axes


def add_mode_derived_parameter(
    samples,
    comparison: CovarianceComparison,
    *,
    mode_index: int,
    dataset: str = "reference",
    name: str | None = None,
    label: str | None = None,
):
    """Return a GetDist sample copy with one generalized mode added as a derived parameter."""

    if not hasattr(samples, "copy") or not hasattr(samples, "addDerived"):
        raise ValueError(
            "add_mode_derived_parameter requires a getdist MCSamples object."
        )
    if mode_index < 1 or mode_index > comparison.mode_vectors.shape[1]:
        raise ValueError(f"Mode index {mode_index} is out of range.")

    derived_name = name or f"mode_{mode_index}"
    derived_label = label or f"m_{{{mode_index}}}"
    projected = project_samples_onto_modes(
        samples,
        comparison,
        dataset=dataset,
        use_normalized_modes=True,
    )

    derived = samples.copy()
    derived.addDerived(projected[:, mode_index - 1], derived_name, label=derived_label)
    return derived


def add_all_mode_derived_parameters(
    samples,
    comparison: CovarianceComparison,
    *,
    dataset: str = "reference",
    name_prefix: str = "mode_",
    label_prefix: str = "m",
):
    """Return a GetDist sample copy with all generalized modes added as derived parameters."""

    if not hasattr(samples, "copy") or not hasattr(samples, "addDerived"):
        raise ValueError(
            "add_all_mode_derived_parameters requires a getdist MCSamples object."
        )

    projected = project_samples_onto_modes(
        samples,
        comparison,
        dataset=dataset,
        use_normalized_modes=True,
    )
    derived = samples.copy()
    for i in range(projected.shape[1]):
        derived.addDerived(
            projected[:, i],
            f"{name_prefix}{i + 1}",
            label=f"{label_prefix}_{{{i + 1}}}",
        )
    return derived


def plot_mode_1d_getdist(
    reference_samples,
    alternative_samples,
    comparison: CovarianceComparison,
    *,
    mode_index: int = 1,
    labels: tuple[str, str] = ("reference", "alternative"),
):
    """Plot one generalized mode as a derived 1D parameter using GetDist."""

    from getdist import plots

    derived_name = f"mode_{mode_index}"
    reference_with_mode = add_mode_derived_parameter(
        reference_samples,
        comparison,
        mode_index=mode_index,
        dataset="reference",
        name=derived_name,
    )
    alternative_with_mode = add_mode_derived_parameter(
        alternative_samples,
        comparison,
        mode_index=mode_index,
        dataset="alternative",
        name=derived_name,
    )

    reference_with_mode.updateSettings({"legend_label": labels[0]})
    alternative_with_mode.updateSettings({"legend_label": labels[1]})

    plotter = plots.get_single_plotter()
    plotter.plot_1d([reference_with_mode, alternative_with_mode], derived_name)
    return plotter, reference_with_mode, alternative_with_mode


def _coerce_samples(
    samples: object,
    *,
    parameter_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray | None]:
    """Convert raw arrays or getdist-like samples into a matrix plus weights."""

    if hasattr(samples, "samples") and hasattr(samples, "getParamNames"):
        names = tuple(samples.getParamNames().list())
        matrix = np.asarray(samples.samples, dtype=float)
        weights = np.asarray(samples.weights, dtype=float)
        return matrix, names, weights

    matrix = np.asarray(samples, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            "Samples must be either a 2D NumPy-like array or a getdist MCSamples object."
        )
    if parameter_names is None:
        raise ValueError(
            "parameter_names must be provided when samples are passed as raw arrays."
        )
    return matrix, tuple(parameter_names), None
