"""Core tools for posterior covariance eigenmode analysis."""

from .api import eigenmodes
from .interpretation import (
    eigenmode_report,
    format_mode_direction,
    format_mode_summary,
    summarize_modes,
    top_correlation_changes,
    top_eigenvector_alignments,
    rotation_subspace_angles,
)
from .modes import analyze_covariances, analyze_covariance
from .plotting import (
    add_all_mode_derived_parameters,
    add_mode_derived_parameter,
    get_mode_samples,
    plot_mode_1d_getdist,
)
from .results import CovarianceAnalysis, CovarianceComparison
from .validation import bootstrap_stability

__all__ = [
    "CovarianceAnalysis",
    "bootstrap_stability",
    "CovarianceComparison",
    "analyze_covariances",
    "analyze_covariance",
    "add_mode_derived_parameter",
    "add_all_mode_derived_parameters",
    "eigenmodes",
    "eigenmode_report",
    "format_mode_direction",
    "format_mode_summary",
    "get_mode_samples",
    "plot_mode_1d_getdist",
    "summarize_modes",
    "top_correlation_changes",
    "top_eigenvector_alignments",
    "rotation_subspace_angles",
]
