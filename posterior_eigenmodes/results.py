from __future__ import annotations

from dataclasses import dataclass, field

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
    rotation_basis: str
    rotation_scales: np.ndarray
    rotation_reference_eigenvalues: np.ndarray
    rotation_alternative_eigenvalues: np.ndarray
    rotation_reference_eigenvectors: np.ndarray
    rotation_alternative_eigenvectors: np.ndarray
    original_eigenvector_overlap: np.ndarray
    numerical_diagnostics: dict[str, float]
    input_covariance_diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        """Geometric mean variance ratio."""
        return float(np.exp(np.mean(np.log(self.degradation_factors))))

    @property
    def A_aniso(self) -> float:
        """Population dispersion of natural-log variance ratios."""
        return float(np.std(np.log(self.degradation_factors)))

    @property
    def shifts(self) -> dict[str, float] | None:
        from .numerics import mahalanobis_shift

        if self.reference_mean is None or self.alternative_mean is None:
            return None
        delta = self.alternative_mean - self.reference_mean
        return {
            "B_in_A": mahalanobis_shift(delta, self.reference_covariance),
            "A_in_B": mahalanobis_shift(delta, self.alternative_covariance),
            "SCCD": mahalanobis_shift(delta, self.reference_covariance + self.alternative_covariance),
        }
