from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import cholesky, polar, solve_triangular
import warnings

from .numerics import covariance_eigensystem, matrix_product, scaled_spd

from .geometry import (
    correlation_matrix,
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


def _normalize_mode_columns(mode_vectors: np.ndarray) -> np.ndarray:
    """Normalize each mode vector to unit Euclidean norm."""

    scale = np.max(np.abs(mode_vectors), axis=0)
    if np.any(scale == 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("Encountered an invalid generalized mode.")
    scaled = mode_vectors / scale
    norms = np.linalg.norm(scaled, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("Encountered a zero-norm generalized mode.")
    return scaled / norms


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

    scales, corr = scaled_spd(matrix, floor=eigenvalue_floor)
    eigenvalues, eigenvectors = covariance_eigensystem(scales, corr)
    standard_deviations = np.sqrt(eigenvalues)

    return CovarianceAnalysis(
        parameter_names=names,
        covariance=matrix,
        correlation=correlation_matrix(matrix, symmetry_atol=symmetry_atol),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        standard_deviations=standard_deviations,
        condition_number=float(eigenvalues[0] / eigenvalues[-1]),
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
    rotation_basis: str = "reference_standardized",
    mean_order: str = "full",
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
        mean_order=mean_order,
    )
    alt_mean = _prepare_optional_mean(
        alternative_mean,
        full_names=tuple(
            parameter_names if alternative_parameter_names is None else alternative_parameter_names
        ),
        selected_names=alt_names,
        label="alternative_mean",
        mean_order=mean_order,
    )

    if ref_matrix.shape != alt_matrix.shape:
        raise ValueError(
            "Reference and alternative covariance selections must have the same dimension."
        )

    ref_matrix = validate_covariance_matrix(ref_matrix, symmetry_atol=symmetry_atol)
    alt_matrix = validate_covariance_matrix(alt_matrix, symmetry_atol=symmetry_atol)
    scales, ref_scaled = scaled_spd(ref_matrix, floor=eigenvalue_floor, label="A")
    alt_scales, alt_corr = scaled_spd(alt_matrix, floor=eigenvalue_floor, label="B")
    reference_eigenvalues, reference_eigenvectors = covariance_eigensystem(scales, ref_scaled)
    alternative_eigenvalues, alternative_eigenvectors = covariance_eigensystem(alt_scales, alt_corr)
    original_overlap = np.clip(
        np.abs(matrix_product(reference_eigenvectors.T, alternative_eigenvectors)), 0, 1
    )
    alt_scaled = alt_matrix / scales[:, None] / scales[None, :]
    if not np.all(np.isfinite(alt_scaled)):
        raise ValueError("Relative covariance scales exceed floating-point range.")

    if rotation_basis == "reference_standardized":
        rot_a, rot_b = ref_scaled, alt_scaled
        rotation_scales = scales
    elif rotation_basis == "original":
        rot_a, rot_b = ref_matrix, alt_matrix
        rotation_scales = np.ones_like(scales)
    else:
        raise ValueError("rotation_basis must be 'reference_standardized' or 'original'.")
    if rotation_basis == "original":
        rot_evals_a, rot_vecs_a = reference_eigenvalues, reference_eigenvectors
        rot_evals_b, rot_vecs_b = alternative_eigenvalues, alternative_eigenvectors
    else:
        rot_evals_a, rot_vecs_a = _symmetric_eigensystem(rot_a)
        rot_evals_b, rot_vecs_b = covariance_eigensystem(alt_scales / scales, alt_corr)
    eigenvector_overlap = np.clip(np.abs(matrix_product(rot_vecs_a.T, rot_vecs_b)), 0, 1)

    # Solve in dimensionless coordinates; never invert the raw covariance.
    L = cholesky(ref_scaled, lower=True)
    left = solve_triangular(L, alt_scaled, lower=True)
    C_chol = solve_triangular(L, left.T, lower=True).T
    C_chol = 0.5 * (C_chol + C_chol.T)
    degradation_factors, U_chol = _symmetric_eigensystem(C_chol)
    if np.any(degradation_factors <= 0) or not np.all(np.isfinite(degradation_factors)):
        raise ValueError("Comparison eigenvalues are not strictly positive and finite; no clipping was applied.")
    V_scaled = solve_triangular(L.T, U_chol, lower=False)
    mode_vectors = V_scaled / scales[:, None]

    # W = Q A^(-1/2): its polar factor converts the Cholesky whitening
    # back to the original symmetric-whitening convention for the displayed C.
    W = solve_triangular(L, np.diag(1.0 / scales), lower=True)
    Q, _ = polar(W)
    whitened_eigenvectors = matrix_product(Q.T, U_chol)
    whitened_alt = matrix_product(matrix_product(Q.T, C_chol), Q)
    whitened_alt = 0.5 * (whitened_alt + whitened_alt.T)
    reference_projection = matrix_product(matrix_product(V_scaled.T, ref_scaled), V_scaled)
    identity_error = np.linalg.norm(reference_projection - np.eye(len(names)), ord=2)
    lhs = matrix_product(alt_scaled, V_scaled)
    rhs = matrix_product(ref_scaled, V_scaled) * degradation_factors
    residual = np.max(np.linalg.norm(lhs - rhs, axis=0) /
                      (np.linalg.norm(lhs, axis=0) + np.linalg.norm(rhs, axis=0)))
    alternative_projection = matrix_product(matrix_product(V_scaled.T, alt_scaled), V_scaled)
    b_error = np.linalg.norm(alternative_projection /
                            np.sqrt(np.outer(degradation_factors, degradation_factors)) - np.eye(len(names)), ord=2)
    diagnostics = {
        "generalized_residual": float(residual),
        "reference_orthogonality_error": float(identity_error),
        "alternative_diagonalization_error": float(b_error),
        "reference_correlation_condition": float(np.linalg.cond(ref_scaled)),
        "alternative_correlation_condition": float(np.linalg.cond(alt_corr)),
    }
    if max(residual, identity_error, b_error) > 1e-7:
        warnings.warn("Large eigensystem residual: inspect numerical_diagnostics before interpretation.", RuntimeWarning)
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
        reference_condition_number=float(reference_eigenvalues[0] / reference_eigenvalues[-1]),
        alternative_condition_number=float(alternative_eigenvalues[0] / alternative_eigenvalues[-1]),
        rotation_basis=rotation_basis,
        rotation_scales=rotation_scales,
        rotation_reference_eigenvalues=rot_evals_a,
        rotation_alternative_eigenvalues=rot_evals_b,
        rotation_reference_eigenvectors=rot_vecs_a,
        rotation_alternative_eigenvectors=rot_vecs_b,
        original_eigenvector_overlap=original_overlap,
        numerical_diagnostics=diagnostics,
    )


def _prepare_optional_mean(
    mean: np.ndarray | None,
    *,
    full_names: tuple[str, ...],
    selected_names: tuple[str, ...],
    label: str,
    mean_order: str = "full",
) -> np.ndarray | None:
    """Align an optional mean vector either from full or already-selected coordinates."""

    if mean_order not in ("full", "selected"):
        raise ValueError("mean_order must be 'full' or 'selected'.")
    if mean is None:
        return None

    mean_array = np.asarray(mean, dtype=float)
    if mean_array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array.")
    if not np.all(np.isfinite(mean_array)):
        raise ValueError(f"{label} contains non-finite values.")

    if mean_order == "selected" and mean_array.shape[0] == len(selected_names):
        return mean_array.copy()

    if mean_array.shape[0] == len(full_names):
        name_to_index = {name: idx for idx, name in enumerate(full_names)}
        return np.asarray(
            [mean_array[name_to_index[name]] for name in selected_names],
            dtype=float,
        )

    if mean_array.shape[0] == len(selected_names):
        return mean_array.copy()

    raise ValueError(
        f"{label} length must match either the full parameter list "
        f"({len(full_names)}) or the selected subspace ({len(selected_names)})."
    )
