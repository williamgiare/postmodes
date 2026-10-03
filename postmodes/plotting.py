from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .api import _coerce_samples
from .numerics import matrix_product
from .results import CovarianceComparison
from .stats import select_sample_columns, weighted_mean


def get_mode_samples(
    reference_samples: object,
    alternative_samples: object,
    comparison: CovarianceComparison,
    *,
    parameter_names_a: Sequence[str] | None = None,
    parameter_names_b: Sequence[str] | None = None,
    center: str | bool = "reference",
    use_normalized_modes: bool | None = None,
    normalization: str = "reference",
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all generalized mode coefficients for the two datasets.

    The returned arrays have shape ``(n_samples, n_modes)`` and can be plotted
    however the user prefers in the notebook.
    """

    reference_center = comparison.reference_mean
    if center == "reference" and reference_center is None:
        matrix, names, weights = _coerce_samples(reference_samples, parameter_names=parameter_names_a)
        selected, _ = select_sample_columns(matrix, names, comparison.parameter_names)
        reference_center = weighted_mean(selected, weights_a if weights_a is not None else weights)
    reference_projection = project_samples_onto_modes(
        reference_samples,
        comparison,
        dataset="reference",
        parameter_names=parameter_names_a,
        center=center,
        use_normalized_modes=use_normalized_modes,
        normalization=normalization,
        reference_center=reference_center,
        weights=weights_a,
    )
    alternative_projection = project_samples_onto_modes(
        alternative_samples,
        comparison,
        dataset="alternative",
        parameter_names=parameter_names_b,
        center=center,
        use_normalized_modes=use_normalized_modes,
        normalization=normalization,
        reference_center=reference_center,
        weights=weights_b,
    )

    return reference_projection, alternative_projection


def project_samples_onto_modes(
    samples: object,
    comparison: CovarianceComparison,
    *,
    dataset: str = "reference",
    parameter_names: Sequence[str] | None = None,
    center: str | bool = "reference",
    use_normalized_modes: bool | None = None,
    normalization: str = "reference",
    reference_center: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Project samples onto the generalized comparison modes.

    Reference normalization gives V.T C_A V = I. Euclidean normalization
    rescales columns, preserving variance ratios but not unit variance in A.
    The default common reference center preserves shifts. ``center='separate'``
    (legacy True) removes them; ``center='none'`` (legacy False) keeps raw values.
    """

    matrix, names, extracted_weights = _coerce_samples(samples, parameter_names=parameter_names)
    weights = extracted_weights if weights is None else weights

    if dataset == "reference":
        requested_names = comparison.parameter_names
    elif dataset == "alternative":
        requested_names = comparison.alternative_parameter_names
    else:
        raise ValueError("dataset must be either 'reference' or 'alternative'.")

    selected, _ = select_sample_columns(matrix, names, requested_names)
    if center is True:
        center = "separate"
    elif center is False:
        center = "none"
    if center == "separate":
        selected = selected - weighted_mean(selected, weights)
    elif center == "reference":
        origin = comparison.reference_mean if reference_center is None else reference_center
        if origin is None and dataset == "reference":
            origin = weighted_mean(selected, weights)
        if origin is None:
            raise ValueError("A common reference_center is required when the comparison has no means; get_mode_samples can infer it from A.")
        origin = np.asarray(origin, dtype=float)
        if origin.shape != (selected.shape[1],) or not np.all(np.isfinite(origin)):
            raise ValueError("reference_center must be a finite vector in selected parameter order.")
        selected = selected - origin
    elif center != "none":
        raise ValueError("center must be 'reference', 'separate', or 'none'.")

    if use_normalized_modes is not None:
        normalization = "euclidean" if use_normalized_modes else "reference"
    if normalization not in ("reference", "euclidean"):
        raise ValueError("normalization must be 'reference' or 'euclidean'.")

    basis = (
        comparison.normalized_mode_coefficients
        if normalization == "euclidean"
        else comparison.mode_vectors
    )
    projection = matrix_product(selected, basis)

    if not np.all(np.isfinite(projection)):
        raise ValueError(
            f"Projection with {normalization} normalization produced non-finite values."
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
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    center: str | bool = "reference",
    normalization: str = "reference",
):
    """Plot 1D histograms of selected generalized mode coefficients."""

    import matplotlib.pyplot as plt

    reference_projection, alternative_projection = get_mode_samples(
        reference_samples,
        alternative_samples,
        comparison,
        parameter_names_a=parameter_names_a,
        parameter_names_b=parameter_names_b,
        weights_a=weights_a,
        weights_b=weights_b,
        center=center,
        normalization=normalization,
    )
    if weights_a is None:
        weights_a = _coerce_samples(reference_samples, parameter_names=parameter_names_a)[2]
    if weights_b is None:
        weights_b = _coerce_samples(alternative_samples, parameter_names=parameter_names_b)[2]

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
        edges = np.histogram_bin_edges(np.concatenate([reference_projection[:, idx], alternative_projection[:, idx]]), bins=bins)
        ax.hist(
            reference_projection[:, idx],
            bins=edges,
            weights=weights_a,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=labels[0],
        )
        ax.hist(
            alternative_projection[:, idx],
            bins=edges,
            weights=weights_b,
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
    center: str | bool = "reference",
    normalization: str = "reference",
    reference_center: np.ndarray | None = None,
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
        center=center,
        normalization=normalization,
        reference_center=reference_center,
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
    center: str | bool = "reference",
    normalization: str = "reference",
    reference_center: np.ndarray | None = None,
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
        center=center,
        normalization=normalization,
        reference_center=reference_center,
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
    center: str | bool = "reference",
    normalization: str = "reference",
):
    """Plot one generalized mode as a derived 1D parameter using GetDist."""

    from getdist import plots

    derived_name = f"mode_{mode_index}"
    matrix, names, weights = _coerce_samples(reference_samples)
    selected, _ = select_sample_columns(matrix, names, comparison.parameter_names)
    origin = comparison.reference_mean
    if origin is None:
        origin = weighted_mean(selected, weights)
    reference_with_mode = add_mode_derived_parameter(
        reference_samples,
        comparison,
        mode_index=mode_index,
        dataset="reference",
        name=derived_name,
        center=center,
        normalization=normalization,
        reference_center=origin,
    )
    alternative_with_mode = add_mode_derived_parameter(
        alternative_samples,
        comparison,
        mode_index=mode_index,
        dataset="alternative",
        name=derived_name,
        center=center,
        normalization=normalization,
        reference_center=origin,
    )

    reference_with_mode.updateSettings({"legend_label": labels[0]})
    alternative_with_mode.updateSettings({"legend_label": labels[1]})

    plotter = plots.get_single_plotter()
    plotter.plot_1d([reference_with_mode, alternative_with_mode], derived_name)
    return plotter, reference_with_mode, alternative_with_mode
