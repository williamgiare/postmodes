from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .results import CovarianceComparison
from .numerics import mahalanobis_shift


@dataclass(frozen=True)
class ParameterContribution:
    """Coefficient of one parameter in a generalized mode."""

    reference_name: str
    alternative_name: str
    raw_coefficient: float
    sigma_scaled_coefficient: float


@dataclass(frozen=True)
class ModeSummary:
    """Summary of one generalized mode."""

    mode_index: int
    degradation_factor: float
    sigma_ratio: float
    status: str
    top_parameters: tuple[ParameterContribution, ...]


@dataclass(frozen=True)
class CorrelationChange:
    """Largest changes in pairwise parameter correlations."""

    parameter_pair_reference: tuple[str, str]
    parameter_pair_alternative: tuple[str, str]
    reference_correlation: float
    alternative_correlation: float
    delta_correlation: float


@dataclass(frozen=True)
class EigenvectorAlignment:
    """Largest alignments between the eigenvector bases of A and B."""

    reference_mode_index: int
    alternative_mode_index: int
    overlap: float


def summarize_modes(
    comparison: CovarianceComparison,
    *,
    top_n_params: int = 3,
    unchanged_tolerance: float = 0.05,
) -> tuple[ModeSummary, ...]:
    """Summarize each generalized mode with its leading parameter coefficients."""

    if top_n_params <= 0:
        raise ValueError("top_n_params must be a positive integer.")

    reference_std = np.sqrt(np.diag(comparison.reference_covariance))
    sigma_scaled = comparison.normalized_mode_coefficients * reference_std[:, None]

    summaries: list[ModeSummary] = []
    for mode_idx, degradation in enumerate(comparison.degradation_factors):
        raw = comparison.normalized_mode_coefficients[:, mode_idx]
        scaled = sigma_scaled[:, mode_idx]
        order = np.argsort(np.abs(raw))[::-1]
        top_indices = order[:top_n_params]

        top_parameters = tuple(
            ParameterContribution(
                reference_name=comparison.parameter_names[i],
                alternative_name=comparison.alternative_parameter_names[i],
                raw_coefficient=float(raw[i]),
                sigma_scaled_coefficient=float(scaled[i]),
            )
            for i in top_indices
        )

        summaries.append(
            ModeSummary(
                mode_index=mode_idx + 1,
                degradation_factor=float(degradation),
                sigma_ratio=float(np.sqrt(max(degradation, 0.0))),
                status=_mode_status(float(degradation), unchanged_tolerance),
                top_parameters=top_parameters,
            )
        )

    return tuple(summaries)


def top_correlation_changes(
    comparison: CovarianceComparison,
    *,
    top_n: int = 5,
) -> tuple[CorrelationChange, ...]:
    """Return the parameter pairs with the largest absolute correlation changes."""

    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    n_dim = len(comparison.parameter_names)
    pairs: list[CorrelationChange] = []
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            pairs.append(
                CorrelationChange(
                    parameter_pair_reference=(
                        comparison.parameter_names[i],
                        comparison.parameter_names[j],
                    ),
                    parameter_pair_alternative=(
                        comparison.alternative_parameter_names[i],
                        comparison.alternative_parameter_names[j],
                    ),
                    reference_correlation=float(comparison.reference_correlation[i, j]),
                    alternative_correlation=float(
                        comparison.alternative_correlation[i, j]
                    ),
                    delta_correlation=float(comparison.correlation_difference[i, j]),
                )
            )

    pairs.sort(key=lambda item: abs(item.delta_correlation), reverse=True)
    return tuple(pairs[:top_n])


def top_eigenvector_alignments(
    comparison: CovarianceComparison,
    *,
    top_n: int = 5,
) -> tuple[EigenvectorAlignment, ...]:
    """Return the largest overlaps between the A and B eigenvector bases."""

    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    overlaps: list[EigenvectorAlignment] = []
    n_modes = comparison.eigenvector_overlap.shape[0]
    for i in range(n_modes):
        for j in range(n_modes):
            overlaps.append(
                EigenvectorAlignment(
                    reference_mode_index=i + 1,
                    alternative_mode_index=j + 1,
                    overlap=float(comparison.eigenvector_overlap[i, j]),
                )
            )

    overlaps.sort(key=lambda item: item.overlap, reverse=True)
    return tuple(overlaps[:top_n])


def rotation_subspace_angles(comparison, reference_modes, alternative_modes):
    """Principal angles in degrees between selected rotation eigenspaces.

    Indices are one-based in the declared rotation basis. Choose whole clusters
    separated from the remaining spectrum. Comparing two complete bases always
    gives zero angles and cannot measure a rotation of axes.
    """
    from scipy.linalg import subspace_angles

    def indices(modes):
        values = tuple(modes)
        n = len(comparison.parameter_names)
        if not values or len(set(values)) != len(values) or any(
            not isinstance(i, (int, np.integer)) or not 1 <= i <= n for i in values
        ):
            raise ValueError("Modes must be unique one-based indices in range.")
        return np.array(values) - 1
    ia, ib = indices(reference_modes), indices(alternative_modes)
    if len(ia) != len(ib):
        raise ValueError("Compared eigenspaces must have equal dimensions.")
    return np.degrees(subspace_angles(
        comparison.rotation_reference_eigenvectors[:, ia],
        comparison.rotation_alternative_eigenvectors[:, ib],
    ))


def format_mode_direction(
    comparison: CovarianceComparison,
    *,
    mode_index: int | None = None,
    max_terms: int | None = None,
    precision: int = 3,
) -> str:
    """Format one or all generalized modes as readable linear combinations."""

    if mode_index is None:
        n_modes = comparison.normalized_mode_coefficients.shape[1]
        return "\n".join(
            f"Mode {idx}: "
            + format_mode_direction(
                comparison,
                mode_index=idx,
                max_terms=max_terms,
                precision=precision,
            )
            for idx in range(1, n_modes + 1)
        )

    coefficients = _mode_coefficients(comparison, mode_index)
    order = np.argsort(np.abs(coefficients))[::-1]
    if max_terms is not None:
        order = order[:max_terms]

    terms = []
    for idx in order:
        coeff = float(coefficients[idx])
        sign = "+" if coeff >= 0.0 else "-"
        magnitude = abs(coeff)
        name_a = comparison.parameter_names[idx]
        name_b = comparison.alternative_parameter_names[idx]
        if name_a == name_b:
            label = name_a
        else:
            label = f"{name_a} ({name_b} in B)"
        terms.append(f"{sign}{magnitude:.{precision}f} {label}")

    if not terms:
        return ""

    formatted = " ".join(terms)
    return formatted[1:] if formatted.startswith("+") else formatted


def format_mode_summary(
    comparison: CovarianceComparison,
    *,
    mode_index: int | None = None,
    max_terms: int | None = None,
    precision: int = 3,
    interpretation: bool = False,
) -> str:
    """Return a compact human-readable summary of one or all generalized modes."""

    if mode_index is None:
        n_modes = comparison.normalized_mode_coefficients.shape[1]
        return "\n\n".join(
            format_mode_summary(
                comparison,
                mode_index=idx,
                max_terms=max_terms,
                precision=precision,
                interpretation=interpretation,
            )
            for idx in range(1, n_modes + 1)
        )

    summaries = summarize_modes(comparison, top_n_params=len(comparison.parameter_names))
    if mode_index < 1 or mode_index > len(summaries):
        raise ValueError(f"Mode index {mode_index} is out of range.")

    summary = summaries[mode_index - 1]
    lines = [
        f"Mode {summary.mode_index}",
        f"rho = {summary.degradation_factor:.{precision}g}",
        f"sigma ratio = {summary.sigma_ratio:.{precision}g}",
        *([f"status = {summary.status}"] if interpretation else []),
        "direction:",
        format_mode_direction(
            comparison,
            mode_index=mode_index,
            max_terms=max_terms,
            precision=precision,
        ),
    ]

    return "\n".join(lines)


def eigenmode_report(
    comparison: CovarianceComparison,
    *,
    max_terms: int | None = None,
    precision: int = 3,
    top_rotation_pairs: int = 6,
    top_correlation_pairs: int = 4,
    interpretation: bool = False,
) -> str:
    """Return a systematic report with sections for A, B, rotations, and C."""

    lines = [
        "A parameters: " + ", ".join(comparison.parameter_names),
        "B parameters: " + ", ".join(comparison.alternative_parameter_names),
        "",
        "A",
        _format_single_posterior_report(
            name="Reference posterior",
            parameter_names=comparison.parameter_names,
            eigenvalues=comparison.reference_eigenvalues,
            eigenvectors=comparison.reference_eigenvectors,
            precision=precision,
            max_terms=max_terms,
        ),
        "",
        "B",
        _format_single_posterior_report(
            name="Alternative posterior",
            parameter_names=comparison.alternative_parameter_names,
            eigenvalues=comparison.alternative_eigenvalues,
            eigenvectors=comparison.alternative_eigenvectors,
            precision=precision,
            max_terms=max_terms,
        ),
        "",
        "Shifts",
        _format_shift_report(
            comparison,
            precision=precision,
        ),
        "",
        "Rotations",
        _format_rotation_report(
            comparison,
            top_n=top_rotation_pairs,
            precision=precision,
        ),
        "",
        "C = C_A^(-1/2) C_B C_A^(-1/2)",
        _format_generalized_report(
            comparison,
            precision=precision,
            max_terms=max_terms,
            top_correlation_pairs=top_correlation_pairs,
        ),
    ]

    report = "\n".join(lines)
    if not interpretation:
        report = "\n".join(line for line in report.splitlines() if not any(
            token in line for token in ("interpretation =", "note =", "axes =", "convention =")
        ))
    return report


def format_comparison_report(
    comparison: CovarianceComparison,
    *,
    max_terms: int | None = None,
    precision: int = 3,
    top_rotation_pairs: int = 6,
    top_correlation_pairs: int = 4,
    interpretation: bool = False,
) -> str:
    """Backward-compatible alias for `eigenmode_report`."""

    return eigenmode_report(
        comparison,
        max_terms=max_terms,
        precision=precision,
        top_rotation_pairs=top_rotation_pairs,
        top_correlation_pairs=top_correlation_pairs,
        interpretation=interpretation,
    )


def _mode_coefficients(
    comparison: CovarianceComparison,
    mode_index: int,
) -> np.ndarray:
    """Return the normalized coefficients for one mode."""

    if mode_index < 1 or mode_index > comparison.normalized_mode_coefficients.shape[1]:
        raise ValueError(f"Mode index {mode_index} is out of range.")
    return comparison.normalized_mode_coefficients[:, mode_index - 1]


def _format_single_posterior_report(
    *,
    name: str,
    parameter_names: tuple[str, ...],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    precision: int,
    max_terms: int | None,
) -> str:
    """Format the eigenmodes of one posterior covariance."""

    lines = [name]
    for idx in range(eigenvalues.shape[0]):
        sigma = float(np.sqrt(max(eigenvalues[idx], 0.0)))
        direction = _format_vector_terms(
            eigenvectors[:, idx],
            parameter_names=parameter_names,
            max_terms=max_terms,
            precision=precision,
        )
        lines.append("")
        lines.append(f"- Mode {idx + 1}")
        lines.append(f"  - lambda = {eigenvalues[idx]:.{precision}g}")
        lines.append(f"  - sigma = {sigma:.{precision}g}")
        lines.append(f"  - direction = {direction}")
    return "\n".join(lines)


def _format_rotation_report(
    comparison: CovarianceComparison,
    *,
    top_n: int,
    precision: int,
) -> str:
    """Format the main alignments between the A and B eigenvector bases."""

    alignments = top_eigenvector_alignments(comparison, top_n=top_n)
    best_overlaps = np.max(comparison.eigenvector_overlap, axis=1)
    lines = [f"- basis = {comparison.rotation_basis}"]
    if comparison.rotation_basis == "reference_standardized":
        lines.append("- axes = PCA of D_A^(-1) C_A D_A^(-1) and D_A^(-1) C_B D_A^(-1); these are distinct from the original PCA modes in sections A/B")
    else:
        lines.append("- axes = original parameter coordinates; overlaps depend on parameter units")
    for item in alignments:
        lines.append(
            f"- A mode {item.reference_mode_index} <-> B mode {item.alternative_mode_index}: "
            f"|overlap| = {item.overlap:.{precision}g}"
        )
    degeneracy_note = _rotation_degeneracy_note(
        comparison.rotation_reference_eigenvalues,
        comparison.rotation_alternative_eigenvalues,
    )
    if degeneracy_note is not None:
        lines.append("- interpretation = individual-axis rotation is ambiguous in the flagged eigenspaces")
        lines.append(f"- note = {degeneracy_note}")
    else:
        lines.append(f"- interpretation = {_rotation_summary(best_overlaps)} (heuristic, in the stated metric)")
    return "\n".join(lines)


def _format_shift_report(
    comparison: CovarianceComparison,
    *,
    precision: int,
) -> str:
    """Format mean-shift diagnostics between A and B."""

    if comparison.reference_mean is None or comparison.alternative_mean is None:
        return "- shift metrics unavailable: mean vectors were not provided"

    delta_mean = comparison.alternative_mean - comparison.reference_mean
    shift_b_in_a = _mahalanobis_shift(
        delta_mean,
        comparison.reference_covariance,
    )
    shift_a_in_b = _mahalanobis_shift(
        delta_mean,
        comparison.alternative_covariance,
    )
    shift_joint = _mahalanobis_shift(
        delta_mean,
        comparison.reference_covariance + comparison.alternative_covariance,
    )

    return "\n".join(
        [
            f"- delta mu = {np.array2string(delta_mean, precision=precision, suppress_small=False)}",
            f"- B shifted from A in A units = {shift_b_in_a:.{precision}g}",
            f"- A shifted from B in B units = {shift_a_in_b:.{precision}g}",
            f"- joint shift in combined units = {shift_joint:.{precision}g}",
            f"- interpretation = {_shift_summary(shift_b_in_a, shift_a_in_b, shift_joint)}",
            "- note = geometric Mahalanobis distances, not Gaussian-sigma tension significances",
        ]
    )


def _format_generalized_report(
    comparison: CovarianceComparison,
    *,
    precision: int,
    max_terms: int | None,
    top_correlation_pairs: int,
) -> str:
    """Format the generalized comparison A -> B."""

    unchanged_tolerance = 0.05
    degradation = comparison.degradation_factors
    sigma_ratios = np.sqrt(np.clip(degradation, 0.0, None))
    alpha_iso = comparison.alpha
    anisotropy = comparison.A_aniso
    n_degraded = int(np.sum(degradation > 1.0 + unchanged_tolerance))
    n_improved = int(np.sum(degradation < 1.0 - unchanged_tolerance))
    n_unchanged = int(
        np.sum(np.abs(degradation - 1.0) <= unchanged_tolerance)
    )
    most_degraded = int(np.argmax(degradation))
    most_improved = int(np.argmin(degradation))

    lines = [
        "constructed comparison matrix C:",
        np.array2string(
            comparison.whitened_comparison_matrix,
            precision=precision,
            suppress_small=False,
        ),
        "",
        "comparison diagnostics:",
        f"- reference condition number = {comparison.reference_condition_number:.{precision}g}",
        f"- alternative condition number = {comparison.alternative_condition_number:.{precision}g}",
        "- note = raw condition numbers depend on units; they do not determine statistical reliability",
        *[f"- {key} = {value:.{precision}g}" for key, value in comparison.numerical_diagnostics.items()],
        "- interpretation = residuals test the numerical solution; chain convergence and sampling uncertainty require separate checks",
        *[f"- {key} = {value:.{precision}g}" for key, value in comparison.input_covariance_diagnostics.items()],
        "",
        "isotropic vs anisotropic deformation:",
        f"- alpha = {alpha_iso:.{precision}g}",
        f"- A_aniso = {anisotropy:.{precision}g}",
        "- convention = alpha is a variance scale; sqrt(alpha) is a linear scale; A_aniso uses natural logs",
        f"- interpretation = {_deformation_summary(alpha_iso, anisotropy)}",
        "",
        "degradation summary:",
        f"- variance ratios rho = {np.array2string(degradation, precision=precision, suppress_small=False)}",
        f"- sigma ratios sqrt(rho) = {np.array2string(sigma_ratios, precision=precision, suppress_small=False)}",
        f"- degraded modes (rho > 1.05) = {n_degraded}",
        f"- improved modes (rho < 0.95) = {n_improved}",
        f"- approximately unchanged modes (|rho - 1| <= 0.05) = {n_unchanged}",
        f"- largest variance ratio: mode {most_degraded + 1} (rho = {degradation[most_degraded]:.{precision}g})",
        f"- smallest variance ratio: mode {most_improved + 1} (rho = {degradation[most_improved]:.{precision}g})",
        "- direction convention = Euclidean-normalized projection coefficients in original units; not parameter importance or ellipsoid displacement axes",
    ]
    for idx in range(comparison.degradation_factors.shape[0]):
        rho = float(comparison.degradation_factors[idx])
        sigma_ratio = float(np.sqrt(max(rho, 0.0)))
        direction = format_mode_direction(
            comparison,
            mode_index=idx + 1,
            max_terms=max_terms,
            precision=precision,
        )
        lines.append("")
        lines.append(f"- Generalized mode {idx + 1}")
        lines.append(f"  - rho = {rho:.{precision}g}")
        lines.append(f"  - sigma ratio = {sigma_ratio:.{precision}g}")
        lines.append(f"  - interpretation = {_comparison_status(rho)}")
        lines.append(f"  - direction = {direction}")

    lines.append("")
    lines.append("largest correlation changes:")
    for item in top_correlation_changes(comparison, top_n=top_correlation_pairs):
        lines.append(
            f"- {item.parameter_pair_reference[0]}-{item.parameter_pair_reference[1]} -> "
            f"{item.parameter_pair_alternative[0]}-{item.parameter_pair_alternative[1]}: "
            f"{item.reference_correlation:.{precision}g} to "
            f"{item.alternative_correlation:.{precision}g} "
            f"(delta = {item.delta_correlation:.{precision}g})"
        )
    return "\n".join(lines)


def _comparison_status(rho: float) -> str:
    """Describe whether B is broader or tighter than A along one generalized mode."""

    if abs(rho - 1.0) <= 0.05:
        return "approximately unchanged relative to A"
    if rho > 1.0:
        return "B is broader than A along this direction"
    return "B is tighter than A along this direction"


def _deformation_summary(alpha_iso: float, anisotropy: float) -> str:
    """Summarize the isotropic and anisotropic parts of the deformation."""

    if abs(alpha_iso - 1.0) <= 0.05:
        iso_text = "little average isotropic rescaling"
    elif alpha_iso > 1.0:
        iso_text = "B is broader than A on average"
    else:
        iso_text = "B is tighter than A on average"

    if anisotropy <= 0.05:
        aniso_text = "and the deformation is close to isotropic"
    else:
        aniso_text = "and the deformation is appreciably anisotropic"

    return f"{iso_text}, {aniso_text}"


def _rotation_summary(best_overlaps: np.ndarray) -> str:
    """Summarize how much the principal bases of A and B rotate."""

    mean_overlap = float(np.mean(best_overlaps))
    min_overlap = float(np.min(best_overlaps))

    if min_overlap >= 0.98:
        return "the principal bases are almost unchanged between A and B"
    if mean_overlap >= 0.9:
        return "the principal bases show only mild rotation or mixing"
    return "the principal bases rotate substantially between A and B"


def _rotation_degeneracy_note(
    reference_eigenvalues: np.ndarray,
    alternative_eigenvalues: np.ndarray,
    *,
    ratio_threshold: float = 1.1,
) -> str | None:
    """Warn when adjacent PCA modes are nearly degenerate in A or B."""

    flagged: list[str] = []

    for label, eigenvalues in (
        ("A", reference_eigenvalues),
        ("B", alternative_eigenvalues),
    ):
        for idx in range(len(eigenvalues) - 1):
            high = float(eigenvalues[idx])
            low = float(eigenvalues[idx + 1])
            if low <= 0.0:
                continue
            ratio = high / low
            if ratio <= ratio_threshold:
                flagged.append(
                    f"{label} modes {idx + 1} and {idx + 2} are nearly degenerate"
                )

    if not flagged:
        return None

    if len(flagged) == 1:
        prefix = flagged[0]
    else:
        prefix = "; ".join(flagged)

    return (
        f"{prefix}, so their individual overlaps are less stable than the overlap "
        "of the corresponding subspace"
    )


def _shift_summary(shift_b_in_a: float, shift_a_in_b: float, shift_joint: float) -> str:
    """Summarize the geometric separation between posterior means."""

    if max(shift_b_in_a, shift_a_in_b, shift_joint) < 1.0:
        return "the posterior means are close compared with the posterior widths"
    if shift_joint < 1.0:
        return (
            "the geometric mean separation is modest, although it can look larger "
            "in the narrower posterior units"
        )
    if shift_a_in_b > 2.0 * shift_b_in_a:
        return "the separation is larger in B units, indicating a tighter B scale along the displacement"
    if shift_b_in_a > 2.0 * shift_a_in_b:
        return "the separation is larger in A units, indicating a tighter A scale along the displacement"
    return "the posterior means show a non-negligible geometric separation"


def _mahalanobis_shift(delta_mean: np.ndarray, covariance: np.ndarray) -> float:
    """Return sqrt(delta^T C^-1 delta) for a positive-definite covariance."""

    return mahalanobis_shift(delta_mean, covariance)


def _format_vector_terms(
    vector: np.ndarray,
    *,
    parameter_names: tuple[str, ...],
    max_terms: int | None,
    precision: int,
) -> str:
    """Format a vector as a readable linear combination."""

    order = np.argsort(np.abs(vector))[::-1]
    if max_terms is not None:
        order = order[:max_terms]

    terms = []
    for idx in order:
        coeff = float(vector[idx])
        sign = "+" if coeff >= 0.0 else "-"
        magnitude = abs(coeff)
        terms.append(f"{sign}{magnitude:.{precision}f} {parameter_names[idx]}")

    if not terms:
        return ""

    formatted = " ".join(terms)
    return formatted[1:] if formatted.startswith("+") else formatted


def _mode_status(degradation_factor: float, unchanged_tolerance: float) -> str:
    """Classify a mode as degraded, improved, or approximately unchanged."""

    if abs(degradation_factor - 1.0) <= unchanged_tolerance:
        return "unchanged"
    if degradation_factor > 1.0:
        return "degraded"
    return "improved"
