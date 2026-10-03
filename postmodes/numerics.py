"""Scale-aware linear algebra shared by comparisons and diagnostics."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular, svd


def matrix_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply 2D arrays without BLAS and reject non-finite results."""
    # Optimized contractions can dispatch to Accelerate and report spurious
    # floating-point exceptions, even when their inputs and outputs are finite.
    product = np.einsum("ik,kj->ij", left, right, optimize=False)
    if not np.all(np.isfinite(product)):
        raise ValueError("Matrix product contains non-finite values.")
    return product


def scaled_spd(matrix, *, floor=1e-14, label="Covariance"):
    """Check positive definiteness in dimensionless correlation coordinates.

    ``floor`` is a relative spectral tolerance, not a regularization strength.
    No eigenvalues are clipped or replaced.
    """
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("eigenvalue_floor must be finite and non-negative.")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0):
        raise ValueError(f"{label} must have strictly positive variances.")
    scales = np.sqrt(diagonal)
    corr = matrix / scales[:, None] / scales[None, :]
    spectrum = np.linalg.eigvalsh(corr)
    if spectrum[0] <= floor * spectrum[-1]:
        raise ValueError(
            f"{label} is singular, indefinite, or unresolved at the requested "
            f"relative tolerance ({floor:g}); min/max correlation eigenvalue "
            f"= {spectrum[0] / spectrum[-1]:.3e}. No regularization was applied."
        )
    return scales, corr


def covariance_eigensystem(scales, corr):
    """PCA from a covariance square root rather than squaring its conditioning.

    Column scaling of the transposed Cholesky factor avoids forming another
    raw covariance in the SVD. Extremely ill-conditioned raw PCA remains
    intrinsically less reliable than the dimensionless comparison.
    """
    root = cholesky(corr, lower=True).T * scales[None, :]
    _, singular_values, vt = svd(root, lapack_driver="gesvd")
    eigenvalues = singular_values ** 2
    if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Raw PCA eigenvalues exceed floating-point range; rescale parameters.")
    return eigenvalues, vt.T


def mahalanobis_shift(delta, covariance):
    """Mahalanobis distance using diagonal scaling and a Cholesky solve."""
    scales, corr = scaled_spd(covariance, floor=0)
    solved = solve_triangular(cholesky(corr, lower=True), delta / scales, lower=True)
    return float(np.linalg.norm(solved))
