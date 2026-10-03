from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .numerics import matrix_product


def weighted_mean(
    samples: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the weighted mean of a sample matrix."""

    matrix = np.asarray(samples, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            "Samples must be a 2D array with shape (n_samples, n_dim); "
            f"received shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Samples contain non-finite values.")

    normalized_weights = _normalized_weights(matrix.shape[0], weights)
    return np.sum(matrix * normalized_weights[:, None], axis=0)


def weighted_covariance(
    samples: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the weighted population covariance of a sample matrix."""

    matrix = np.asarray(samples, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            "Samples must be a 2D array with shape (n_samples, n_dim); "
            f"received shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Samples contain non-finite values.")

    normalized_weights = _normalized_weights(matrix.shape[0], weights)
    mean = weighted_mean(matrix, normalized_weights)
    centered = matrix - mean
    covariance = matrix_product(
        centered.T,
        centered * normalized_weights[:, None],
    )
    covariance = 0.5 * (covariance + covariance.T)

    if not np.all(np.isfinite(covariance)):
        raise ValueError("Weighted covariance contains non-finite values.")

    return covariance


def select_sample_columns(
    samples: np.ndarray,
    parameter_names: Sequence[str],
    selected_parameters: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract sample columns in the requested parameter order."""

    matrix = np.asarray(samples, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            "Samples must be a 2D array with shape (n_samples, n_dim); "
            f"received shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Samples contain non-finite values.")

    names = tuple(parameter_names)
    if len(names) != matrix.shape[1]:
        raise ValueError(
            "Number of parameter names must match sample dimension; "
            f"received {len(names)} names for dimension {matrix.shape[1]}."
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
            "Requested parameters were not found in the samples: "
            + ", ".join(missing)
        )

    indices = np.array([names.index(name) for name in requested], dtype=int)
    return matrix[:, indices], requested


def _normalized_weights(
    n_samples: int,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Return non-negative weights normalized to unit sum."""

    if n_samples <= 0:
        raise ValueError("At least one sample is required.")

    if weights is None:
        return np.full(n_samples, 1.0 / n_samples)

    array = np.asarray(weights, dtype=float)
    if array.shape != (n_samples,):
        raise ValueError(
            "Weights must have shape (n_samples,); "
            f"received shape {array.shape} for {n_samples} samples."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Weights contain non-finite values.")
    if np.any(array < 0.0):
        raise ValueError("Weights must be non-negative.")

    largest = float(np.max(array))
    if largest <= 0.0:
        raise ValueError("Weights must have strictly positive sum.")
    scaled = array / largest
    return scaled / np.sum(scaled)
