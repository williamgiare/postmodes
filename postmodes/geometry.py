from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_covariance_matrix(
    covariance: np.ndarray,
    *,
    symmetry_atol: float = 1e-10,
    require_finite: bool = True,
) -> np.ndarray:
    """Return a validated covariance matrix as a float array.
    
    Symmetry is tested in dimensionless diagonal-scaled coordinates when
    variances are positive. ``symmetry_atol`` is then unit independent.
    """

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(
            "Covariance matrix must be a square 2D array; "
            f"received shape {matrix.shape}."
        )

    if require_finite and not np.all(np.isfinite(matrix)):
        raise ValueError("Covariance matrix contains non-finite values.")

    if not np.isfinite(symmetry_atol) or symmetry_atol < 0:
        raise ValueError("symmetry_atol must be finite and non-negative.")
    scales = np.sqrt(np.abs(np.diag(matrix)))
    checked = matrix / scales[:, None] / scales[None, :] if np.all(scales > 0) else matrix
    if not np.allclose(checked, checked.T, atol=symmetry_atol, rtol=0.0):
        max_asymmetry = float(np.max(np.abs(matrix - matrix.T)))
        raise ValueError(
            "Covariance matrix must be symmetric within tolerance; "
            f"maximum asymmetry is {max_asymmetry:.3e}."
        )

    return 0.5 * matrix + 0.5 * matrix.T


def covariance_condition_number(
    covariance: np.ndarray,
    *,
    symmetry_atol: float = 1e-10,
) -> float:
    """Return the spectral condition number of a validated covariance matrix."""

    matrix = validate_covariance_matrix(covariance, symmetry_atol=symmetry_atol)
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_abs_eigenvalue = float(np.min(np.abs(eigenvalues)))
    max_abs_eigenvalue = float(np.max(np.abs(eigenvalues)))

    if min_abs_eigenvalue == 0.0:
        return float("inf")

    return max_abs_eigenvalue / min_abs_eigenvalue


def correlation_matrix(
    covariance: np.ndarray,
    *,
    symmetry_atol: float = 1e-10,
) -> np.ndarray:
    """Convert a covariance matrix into a correlation matrix."""

    matrix = validate_covariance_matrix(covariance, symmetry_atol=symmetry_atol)
    variances = np.diag(matrix)

    if np.any(variances <= 0.0):
        raise ValueError(
            "Correlation matrix is only defined for strictly positive variances."
        )

    std = np.sqrt(variances)
    outer = np.outer(std, std)
    correlation = matrix / outer
    np.fill_diagonal(correlation, 1.0)
    return correlation


def select_parameter_subspace(
    covariance: np.ndarray,
    parameter_names: Sequence[str],
    selected_parameters: Sequence[str] | None = None,
    *,
    symmetry_atol: float = 1e-10,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract a covariance submatrix in the user-requested parameter order."""

    matrix = validate_covariance_matrix(covariance, symmetry_atol=symmetry_atol)
    names = tuple(parameter_names)

    if len(names) != matrix.shape[0]:
        raise ValueError(
            "Number of parameter names must match covariance dimension; "
            f"received {len(names)} names for dimension {matrix.shape[0]}."
        )

    if len(set(names)) != len(names):
        raise ValueError("Parameter names must be unique.")

    if selected_parameters is None:
        return matrix.copy(), names

    requested = tuple(selected_parameters)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Selected parameters must be non-empty and unique.")
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(
            "Requested parameters were not found in the covariance: "
            + ", ".join(missing)
        )

    indices = np.array([names.index(name) for name in requested], dtype=int)
    submatrix = matrix[np.ix_(indices, indices)]
    return submatrix, requested
