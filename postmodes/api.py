from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from dataclasses import replace
import warnings

import numpy as np

from .io import load_covmat, load_getdist_chain
from .geometry import select_parameter_subspace
from .modes import analyze_covariances
from .results import CovarianceComparison
from .stats import select_sample_columns, weighted_covariance, weighted_mean


def eigenmodes(
    chain_root_a: str | Path | None = None,
    chain_root_b: str | Path | None = None,
    *,
    params_A: Sequence[str] | None = None,
    params_B: Sequence[str] | None = None,
    params: Sequence[str] | None = None,
    covmat_a: str | Path | None = None,
    covmat_b: str | Path | None = None,
    chain_settings: dict[str, Any] | None = None,
    symmetry_atol: float = 1e-10,
    eigenvalue_floor: float = 1e-14,
    rotation_basis: str = "reference_standardized",
    check_covariances: bool = False,
    covariance_check_tolerance: float = 0.1,
) -> CovarianceComparison:
    """High-level posterior comparison using the best available input information.

    Supported modes:
    - only `covmat_a` and `covmat_b`: use covariance matrices directly
    - only `chain_root_a` and `chain_root_b`: reconstruct covariances and means from chains
    - both covariance paths and chain paths: use covariance files for the covariance analysis,
      and use the chains only to estimate posterior means for the shift diagnostics
    """

    _validate_path_pair(
        chain_root_a,
        chain_root_b,
        label_a="chain_root_a",
        label_b="chain_root_b",
    )
    _validate_path_pair(
        covmat_a,
        covmat_b,
        label_a="covmat_a",
        label_b="covmat_b",
    )

    have_chains = chain_root_a is not None and chain_root_b is not None
    have_covmats = covmat_a is not None and covmat_b is not None

    if not have_chains and not have_covmats:
        raise ValueError(
            "eigenmodes requires either both covmat paths, both chain roots, or both."
        )

    mean_a = None
    mean_b = None
    if not np.isfinite(covariance_check_tolerance) or covariance_check_tolerance < 0:
        raise ValueError("covariance_check_tolerance must be finite and non-negative.")
    if have_chains:
        samples_a = load_getdist_chain(chain_root_a, settings=chain_settings)
        samples_b = load_getdist_chain(chain_root_b, settings=chain_settings)
    if have_covmats:
        covariance_a, names_a = load_covmat(covmat_a)
        covariance_b, names_b = load_covmat(covmat_b)
    else:
        names_a = tuple(p.name for p in samples_a.getParamNames().names if not p.isDerived)
        names_b = tuple(p.name for p in samples_b.getParamNames().names if not p.isDerived)

    if params is not None:
        if params_A is not None or params_B is not None:
            raise ValueError("Use either params or params_A/params_B, not both.")
        params_A = params_B = params
    if params_A is None and params_B is None:
        params_A = tuple(n for n in names_a if n in names_b)
        params_B = params_A
        excluded = set(names_a).symmetric_difference(names_b)
        if excluded:
            warnings.warn("Automatic selection excludes unmatched names: " + ", ".join(sorted(excluded)) + ". Use params_A/params_B for explicit mappings.", UserWarning)
    elif params_A is None:
        params_A = params_B
    elif params_B is None:
        params_B = params_A
    if len(params_A) == 0 or len(params_A) != len(params_B):
        raise ValueError("Parameter selections must be non-empty and have equal lengths.")

    if have_covmats and have_chains:
        selected_a, _ = select_sample_columns(samples_a.samples, samples_a.getParamNames().list(), params_A)
        selected_b, _ = select_sample_columns(samples_b.samples, samples_b.getParamNames().list(), params_B)
        mean_a = weighted_mean(selected_a, samples_a.weights)
        mean_b = weighted_mean(selected_b, samples_b.weights)

    if have_covmats:

        aligned_a, selected_names_a = select_parameter_subspace(
            covariance_a,
            names_a,
            selected_parameters=params_A,
            symmetry_atol=symmetry_atol,
        )
        aligned_b, selected_names_b = select_parameter_subspace(
            covariance_b,
            names_b,
            selected_parameters=params_B,
            symmetry_atol=symmetry_atol,
        )

        result = compare_covariances(
            aligned_a,
            aligned_b,
            selected_names_a,
            mean_a=mean_a,
            mean_b=mean_b,
            alternative_parameter_names=selected_names_b,
            symmetry_atol=symmetry_atol,
            eigenvalue_floor=eigenvalue_floor,
            rotation_basis=rotation_basis,
        )
        if have_chains and check_covariances:
            diagnostics = {}
            for label, supplied, data, weights in (
                ("A", aligned_a, selected_a, samples_a.weights),
                ("B", aligned_b, selected_b, samples_b.weights),
            ):
                estimated = weighted_covariance(data, weights)
                scales = np.sqrt(np.diag(supplied))
                baseline = supplied / scales[:, None] / scales[None, :]
                difference = (estimated - supplied) / scales[:, None] / scales[None, :]
                error = float(np.linalg.norm(difference) / np.linalg.norm(baseline))
                diagnostics[f"{label}_standardized_relative_covariance_difference"] = error
                if error > covariance_check_tolerance:
                    warnings.warn(f"{label}: supplied covariance differs from the weighted chain estimate ({error:.3g}).", UserWarning)
            result = replace(result, input_covariance_diagnostics=diagnostics)
        return result

    return compare_samples(
        samples_a,
        samples_b,
        selected_parameters=params_A,
        selected_parameters_b=params_B,
        symmetry_atol=symmetry_atol,
        eigenvalue_floor=eigenvalue_floor,
        rotation_basis=rotation_basis,
    )


def compare_covariances(
    covariance_a: np.ndarray,
    covariance_b: np.ndarray,
    parameter_names: Sequence[str],
    *,
    mean_a: np.ndarray | None = None,
    mean_b: np.ndarray | None = None,
    alternative_parameter_names: Sequence[str] | None = None,
    selected_parameters: Sequence[str] | None = None,
    selected_parameters_b: Sequence[str] | None = None,
    symmetry_atol: float = 1e-10,
    eigenvalue_floor: float = 1e-14,
    rotation_basis: str = "reference_standardized",
    mean_order: str = "full",
) -> CovarianceComparison:
    """High-level entry point for direct covariance comparison."""

    return analyze_covariances(
        covariance_a,
        covariance_b,
        parameter_names,
        reference_mean=mean_a,
        alternative_mean=mean_b,
        alternative_parameter_names=alternative_parameter_names,
        selected_parameters=selected_parameters,
        selected_parameters_b=selected_parameters_b,
        symmetry_atol=symmetry_atol,
        eigenvalue_floor=eigenvalue_floor,
        rotation_basis=rotation_basis,
        mean_order=mean_order,
    )


def compare_samples(
    samples_a: object,
    samples_b: object,
    *,
    parameter_names: Sequence[str] | None = None,
    parameter_names_b: Sequence[str] | None = None,
    selected_parameters: Sequence[str] | None = None,
    selected_parameters_b: Sequence[str] | None = None,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    symmetry_atol: float = 1e-10,
    eigenvalue_floor: float = 1e-14,
    rotation_basis: str = "reference_standardized",
) -> CovarianceComparison:
    """High-level entry point for sample-based posterior comparison."""

    matrix_a, names_a, extracted_weights_a = _coerce_samples(
        samples_a,
        parameter_names=parameter_names,
    )
    matrix_b, names_b, extracted_weights_b = _coerce_samples(
        samples_b,
        parameter_names=parameter_names if parameter_names_b is None else parameter_names_b,
    )

    selected_a, selected_names = select_sample_columns(
        matrix_a,
        names_a,
        selected_parameters=selected_parameters,
    )
    selected_b, selected_names_b = select_sample_columns(
        matrix_b,
        names_b,
        selected_parameters=selected_parameters
        if selected_parameters_b is None
        else selected_parameters_b,
    )

    if selected_a.shape[1] != selected_b.shape[1]:
        raise ValueError(
            "Selected parameter lists must define the same number of physical parameters."
        )

    covariance_a = weighted_covariance(
        selected_a,
        weights=weights_a if weights_a is not None else extracted_weights_a,
    )
    covariance_b = weighted_covariance(
        selected_b,
        weights=weights_b if weights_b is not None else extracted_weights_b,
    )
    mean_a = weighted_mean(
        selected_a,
        weights=weights_a if weights_a is not None else extracted_weights_a,
    )
    mean_b = weighted_mean(
        selected_b,
        weights=weights_b if weights_b is not None else extracted_weights_b,
    )

    return compare_covariances(
        covariance_a,
        covariance_b,
        selected_names,
        mean_a=mean_a,
        mean_b=mean_b,
        alternative_parameter_names=selected_names_b,
        symmetry_atol=symmetry_atol,
        eigenvalue_floor=eigenvalue_floor,
        rotation_basis=rotation_basis,
    )


def compare_covmats(
    covmat_a: str | Path,
    covmat_b: str | Path,
    *,
    params_A: Sequence[str],
    params_B: Sequence[str],
    symmetry_atol: float = 1e-10,
    eigenvalue_floor: float = 1e-14,
) -> CovarianceComparison:
    """Compare two Cobaya covariance-matrix files by explicit parameter lists.

    `params_A[i]` and `params_B[i]` are assumed to be the same physical
    parameter, even if their names differ across the two files.
    """

    return eigenmodes(
        covmat_a=covmat_a,
        covmat_b=covmat_b,
        params_A=params_A,
        params_B=params_B,
        symmetry_atol=symmetry_atol,
        eigenvalue_floor=eigenvalue_floor,
    )


def compare_MCMC_chains(
    chain_root_a: str | Path,
    chain_root_b: str | Path,
    *,
    params_A: Sequence[str],
    params_B: Sequence[str],
    chain_settings: dict[str, Any] | None = None,
    symmetry_atol: float = 1e-10,
    eigenvalue_floor: float = 1e-14,
) -> CovarianceComparison:
    """Load two Cobaya/GetDist chains and compare aligned physical parameters.

    `params_A[i]` and `params_B[i]` are assumed to be the same physical
    parameter, even if their names differ across the two chains.
    """

    return eigenmodes(
        chain_root_a=chain_root_a,
        chain_root_b=chain_root_b,
        params_A=params_A,
        params_B=params_B,
        chain_settings=chain_settings,
        symmetry_atol=symmetry_atol,
        eigenvalue_floor=eigenvalue_floor,
    )


def _coerce_samples(
    samples: object,
    *,
    parameter_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray | None]:
    """Convert raw arrays or getdist MCSamples into a sample matrix."""

    if hasattr(samples, "samples") and hasattr(samples, "weights"):
        if hasattr(samples, "getParamNames"):
            names = tuple(samples.getParamNames().list())
        elif parameter_names is not None:
            names = tuple(parameter_names)
        else:
            raise ValueError("parameter_names are required for GetDist WeightedSamples without metadata.")
        matrix = np.asarray(samples.samples, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(
                "GetDist samples must expose a 2D sample matrix; "
                f"received shape {matrix.shape}."
            )
        if matrix.shape[1] != len(names):
            raise ValueError(
                "Number of GetDist parameter names must match the sample dimension; "
                f"received {len(names)} names for dimension {matrix.shape[1]}."
            )
        weights = None if samples.weights is None else np.asarray(samples.weights, dtype=float)
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


def _validate_path_pair(
    path_a: str | Path | None,
    path_b: str | Path | None,
    *,
    label_a: str,
    label_b: str,
) -> None:
    """Ensure paired inputs are either both provided or both omitted."""

    if (path_a is None) != (path_b is None):
        raise ValueError(f"{label_a} and {label_b} must be provided together.")
