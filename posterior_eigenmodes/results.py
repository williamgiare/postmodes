from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CovarianceAnalysis:
    """Results of the eigenmode analysis for a single covariance matrix."""

    parameter_names: tuple[str, ...]
    covariance: np.ndarray
    correlation: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    standard_deviations: np.ndarray
    condition_number: float


@dataclass(frozen=True)
class CovarianceComparison:
    """Results of the generalized eigenmode comparison between two covariances."""

    parameter_names: tuple[str, ...]
    alternative_parameter_names: tuple[str, ...]
    reference_mean: np.ndarray | None
    alternative_mean: np.ndarray | None
    reference_covariance: np.ndarray
    alternative_covariance: np.ndarray
    reference_correlation: np.ndarray
    alternative_correlation: np.ndarray
    correlation_difference: np.ndarray
    whitened_comparison_matrix: np.ndarray
    degradation_factors: np.ndarray
    whitened_eigenvectors: np.ndarray
    mode_vectors: np.ndarray
    normalized_mode_coefficients: np.ndarray
    reference_eigenvalues: np.ndarray
    reference_eigenvectors: np.ndarray
    alternative_eigenvalues: np.ndarray
    alternative_eigenvectors: np.ndarray
    eigenvector_overlap: np.ndarray
    reference_condition_number: float
    alternative_condition_number: float
