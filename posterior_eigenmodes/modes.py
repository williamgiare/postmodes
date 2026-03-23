from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .geometry import (
    correlation_matrix,
    covariance_condition_number,
    select_parameter_subspace,
    validate_covariance_matrix,
)
from .results import CovarianceAnalysis
from .results import CovarianceComparison


def _symmetric_eigensystem(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and eigenvectors sorted from largest to smallest."""

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _inverse_symmetric_square_root(
    matrix: np.ndarray,
    *,
    symmetry_atol: float = 1e-10,
    eigenvalue_floor: float = 1e-14,
) -> np.ndarray:
    """Compute a stable inverse square root for a symmetric positive matrix."""

    validated = validate_covariance_matrix(matrix, symmetry_atol=symmetry_atol)
    eigenvalues, eigenvectors = np.linalg.eigh(validated)

    if np.any(eigenvalues <= eigenvalue_floor):
        min_eigenvalue = float(np.min(eigenvalues))
        raise ValueError(
            "Reference covariance is not positive definite enough for whitening; "
            f"minimum eigenvalue is {min_eigenvalue:.3e}, "
            f"required floor is {eigenvalue_floor:.3e}."
        )

    inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues)
    return (eigenvectors * inv_sqrt_eigenvalues) @ eigenvectors.T


def _normalize_mode_columns(mode_vectors: np.ndarray) -> np.ndarray:
    """Normalize each mode vector to unit Euclidean norm."""

    norms = np.linalg.norm(mode_vectors, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("Encountered a zero-norm generalized mode.")
    return mode_vectors / norms


def analyze_covariance(
    covariance: np.ndarray,
    parameter_names: Sequence[str],
    *,
    selected_parameters: Sequence[str] | None = None,
    symmetry_atol: float = 1e-10,
    eigenvalue_floor: float = 0.0,
) -> CovarianceAnalysis:
    """Analyze a single covariance matrix through its symmetric eigenmodes."""

    matrix, names = select_parameter_subspace(
        covariance,
        parameter_names,
        selected_parameters,
        symmetry_atol=symmetry_atol,
    )
    matrix = validate_covariance_matrix(matrix, symmetry_atol=symmetry_atol)

    eigenvalues, eigenvectors = _symmetric_eigensystem(matrix)

    if np.any(eigenvalues < eigenvalue_floor):
        min_eigenvalue = float(np.min(eigenvalues))
        raise ValueError(
            "Covariance matrix has eigenvalues below the allowed floor; "
            f"minimum eigenvalue is {min_eigenvalue:.3e}, "
            f"floor is {eigenvalue_floor:.3e}."
        )

    clipped = np.clip(eigenvalues, a_min=0.0, a_max=None)
    standard_deviations = np.sqrt(clipped)

    return CovarianceAnalysis(
        parameter_names=names,
        covariance=matrix,
        correlation=correlation_matrix(matrix, symmetry_atol=symmetry_atol),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        standard_deviations=standard_deviations,
        condition_number=covariance_condition_number(
            matrix, symmetry_atol=symmetry_atol
        ),
    )


def analyze_covariances(
    reference_covariance: np.ndarray,
    alternative_covariance: np.ndarray,
    parameter_names: Sequence[str],
    *,
    reference_mean: np.ndarray | None = None,
    alternative_mean: np.ndarray | None = None,
    alternative_parameter_names: Sequence[str] | None = None,
    selected_parameters: Sequence[str] | None = None,
    selected_parameters_b: Sequence[str] | None = None,
    symmetry_atol: float = 1e-10,
    eigenvalue_floor: float = 1e-14,
) -> CovarianceComparison:
    """Compare two covariances through a whitened generalized eigenmode analysis."""

    ref_matrix, names = select_parameter_subspace(
        reference_covariance,
        parameter_names,
        selected_parameters,
        symmetry_atol=symmetry_atol,
    )
    alt_matrix, alt_names = select_parameter_subspace(
        alternative_covariance,
        parameter_names if alternative_parameter_names is None else alternative_parameter_names,
        selected_parameters if selected_parameters_b is None else selected_parameters_b,
        symmetry_atol=symmetry_atol,
    )

    ref_mean = _prepare_optional_mean(
        reference_mean,
        full_names=tuple(parameter_names),
        selected_names=names,
        label="reference_mean",
    )
    alt_mean = _prepare_optional_mean(
        alternative_mean,
        full_names=tuple(
            parameter_names if alternative_parameter_names is None else alternative_parameter_names
        ),
        selected_names=alt_names,
        label="alternative_mean",
    )

    if ref_matrix.shape != alt_matrix.shape:
        raise ValueError(
            "Reference and alternative covariance selections must have the same dimension."
        )

    ref_matrix = validate_covariance_matrix(ref_matrix, symmetry_atol=symmetry_atol)
    alt_matrix = validate_covariance_matrix(alt_matrix, symmetry_atol=symmetry_atol)
    reference_eigenvalues, reference_eigenvectors = _symmetric_eigensystem(ref_matrix)
    alternative_eigenvalues, alternative_eigenvectors = _symmetric_eigensystem(alt_matrix)
    eigenvector_overlap = np.abs(reference_eigenvectors.T @ alternative_eigenvectors)

    ref_inv_sqrt = _inverse_symmetric_square_root(
        ref_matrix,
        symmetry_atol=symmetry_atol,
        eigenvalue_floor=eigenvalue_floor,
    )
    whitened_alt = ref_inv_sqrt @ alt_matrix @ ref_inv_sqrt
    whitened_alt = 0.5 * (whitened_alt + whitened_alt.T)

    degradation_factors, whitened_eigenvectors = _symmetric_eigensystem(whitened_alt)

    if np.any(degradation_factors < -eigenvalue_floor):
        min_eigenvalue = float(np.min(degradation_factors))
        raise ValueError(
            "Whitened comparison matrix has significantly negative eigenvalues; "
            f"minimum eigenvalue is {min_eigenvalue:.3e}."
        )

    degradation_factors = np.clip(degradation_factors, a_min=0.0, a_max=None)
    mode_vectors = ref_inv_sqrt @ whitened_eigenvectors
    normalized_mode_coefficients = _normalize_mode_columns(mode_vectors)

    reference_correlation = correlation_matrix(ref_matrix, symmetry_atol=symmetry_atol)
    alternative_correlation = correlation_matrix(
        alt_matrix, symmetry_atol=symmetry_atol
    )

    return CovarianceComparison(
        parameter_names=names,
        alternative_parameter_names=alt_names,
        reference_mean=ref_mean,
        alternative_mean=alt_mean,
        reference_covariance=ref_matrix,
        alternative_covariance=alt_matrix,
        reference_correlation=reference_correlation,
        alternative_correlation=alternative_correlation,
        correlation_difference=alternative_correlation - reference_correlation,
        whitened_comparison_matrix=whitened_alt,
        degradation_factors=degradation_factors,
        whitened_eigenvectors=whitened_eigenvectors,
        mode_vectors=mode_vectors,
        normalized_mode_coefficients=normalized_mode_coefficients,
        reference_eigenvalues=reference_eigenvalues,
        reference_eigenvectors=reference_eigenvectors,
        alternative_eigenvalues=alternative_eigenvalues,
        alternative_eigenvectors=alternative_eigenvectors,
        eigenvector_overlap=eigenvector_overlap,
        reference_condition_number=covariance_condition_number(
            ref_matrix, symmetry_atol=symmetry_atol
        ),
        alternative_condition_number=covariance_condition_number(
            alt_matrix, symmetry_atol=symmetry_atol
        ),
    )


def _prepare_optional_mean(
    mean: np.ndarray | None,
    *,
    full_names: tuple[str, ...],
    selected_names: tuple[str, ...],
    label: str,
) -> np.ndarray | None:
    """Align an optional mean vector either from full or already-selected coordinates."""

    if mean is None:
        return None

    mean_array = np.asarray(mean, dtype=float)
    if mean_array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array.")
    if not np.all(np.isfinite(mean_array)):
        raise ValueError(f"{label} contains non-finite values.")

    if mean_array.shape[0] == len(selected_names):
        return mean_array.copy()

    if mean_array.shape[0] == len(full_names):
        name_to_index = {name: idx for idx, name in enumerate(full_names)}
        return np.asarray(
            [mean_array[name_to_index[name]] for name in selected_names],
            dtype=float,
        )

    raise ValueError(
        f"{label} length must match either the full parameter list "
        f"({len(full_names)}) or the selected subspace ({len(selected_names)})."
    )
