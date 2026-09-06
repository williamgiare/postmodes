from __future__ import annotations

from pathlib import Path

import numpy as np
from getdist import MCSamples

from postmodes import (
    add_all_mode_derived_parameters,
    add_mode_derived_parameter,
    analyze_covariance,
    analyze_covariances,
    eigenmodes,
    eigenmode_report,
    format_mode_direction,
    format_mode_summary,
    get_mode_samples,
    plot_mode_1d_getdist,
    summarize_modes,
    top_correlation_changes,
    top_eigenvector_alignments,
)
from postmodes.api import (
    compare_MCMC_chains,
    compare_covariances,
    compare_covmats,
    compare_samples,
)
from postmodes.geometry import select_parameter_subspace
from postmodes.interpretation import format_comparison_report
from postmodes.io import load_covmat
from postmodes.plotting import project_samples_onto_modes
from postmodes.stats import weighted_covariance


def test_select_parameter_subspace_respects_requested_order() -> None:
    covariance = np.array(
        [
            [4.0, 1.0, 0.2],
            [1.0, 3.0, 0.5],
            [0.2, 0.5, 2.0],
        ]
    )
    names = ["omega_m", "sigma8", "h"]

    subcov, subnames = select_parameter_subspace(
        covariance,
        names,
        selected_parameters=["h", "omega_m"],
    )

    expected = np.array(
        [
            [2.0, 0.2],
            [0.2, 4.0],
        ]
    )
    assert subnames == ("h", "omega_m")
    np.testing.assert_allclose(subcov, expected)


def test_analyze_covariance_returns_sorted_eigenmodes() -> None:
    covariance = np.array(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ]
    )

    result = analyze_covariance(covariance, ["omega_m", "sigma8"])

    assert result.parameter_names == ("omega_m", "sigma8")
    assert result.eigenvalues.shape == (2,)
    assert result.eigenvectors.shape == (2, 2)
    assert result.eigenvalues[0] >= result.eigenvalues[1]
    np.testing.assert_allclose(result.correlation.diagonal(), np.ones(2))


def test_analyze_covariance_rejects_asymmetric_input() -> None:
    covariance = np.array(
        [
            [1.0, 0.3],
            [0.1, 2.0],
        ]
    )

    try:
        analyze_covariance(covariance, ["omega_m", "sigma8"])
    except ValueError as exc:
        assert "symmetric" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for asymmetric covariance.")


def test_analyze_covariances_recovers_simple_degradation_factors() -> None:
    reference = np.array(
        [
            [4.0, 0.0],
            [0.0, 1.0],
        ]
    )
    alternative = np.array(
        [
            [8.0, 0.0],
            [0.0, 0.5],
        ]
    )

    result = analyze_covariances(reference, alternative, ["omega_m", "sigma8"])

    np.testing.assert_allclose(result.degradation_factors, np.array([2.0, 0.5]))
    np.testing.assert_allclose(
        np.linalg.norm(result.normalized_mode_coefficients, axis=0),
        np.ones(2),
    )
    assert result.alternative_parameter_names == ("omega_m", "sigma8")


def test_compare_covariances_matches_low_level_analysis() -> None:
    reference = np.array([[4.0, 0.0], [0.0, 1.0]])
    alternative = np.array([[8.0, 0.0], [0.0, 0.5]])

    direct = compare_covariances(reference, alternative, ["omega_m", "sigma8"])
    low_level = analyze_covariances(reference, alternative, ["omega_m", "sigma8"])

    np.testing.assert_allclose(direct.degradation_factors, low_level.degradation_factors)


def test_compare_covariances_accepts_selected_mean_vectors_with_different_alt_names() -> None:
    reference = np.array([[4.0, 0.0], [0.0, 1.0]])
    alternative = np.array([[8.0, 0.0], [0.0, 0.5]])

    result = compare_covariances(
        reference,
        alternative,
        ["theta_s_1e2", "omega_b"],
        mean_a=np.array([1.0, 2.0]),
        mean_b=np.array([1.5, 2.5]),
        alternative_parameter_names=["theta_s_100", "omega_b"],
    )

    np.testing.assert_allclose(result.reference_mean, np.array([1.0, 2.0]))
    np.testing.assert_allclose(result.alternative_mean, np.array([1.5, 2.5]))


def test_weighted_covariance_for_raw_samples() -> None:
    samples = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 2.0],
        ]
    )
    weights = np.array([1.0, 2.0, 1.0])

    covariance = weighted_covariance(samples, weights)
    expected = np.array(
        [
            [2.0, 1.0],
            [1.0, 0.75],
        ]
    )
    np.testing.assert_allclose(covariance, expected)


def test_compare_samples_accepts_getdist_samples() -> None:
    samples_a = MCSamples(
        samples=np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [4.0, 2.0],
            ]
        ),
        names=["omega_m", "sigma8"],
        labels=["omega_m", "sigma8"],
    )
    samples_b = MCSamples(
        samples=np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [8.0, 1.0],
            ]
        ),
        names=["omega_m", "sigma8"],
        labels=["omega_m", "sigma8"],
    )

    result = compare_samples(
        samples_a,
        samples_b,
        selected_parameters=["omega_m", "sigma8"],
        selected_parameters_b=["omega_m", "sigma8"],
    )

    assert result.parameter_names == ("omega_m", "sigma8")
    assert result.alternative_parameter_names == ("omega_m", "sigma8")
    assert result.degradation_factors.shape == (2,)


def test_load_covmat_reads_header_and_matrix(tmp_path: Path) -> None:
    covmat_path = tmp_path / "test.covmat"
    covmat_path.write_text(
        "# omega_m sigma8 h\n"
        "4.0 1.0 0.2\n"
        "1.0 3.0 0.5\n"
        "0.2 0.5 2.0\n",
        encoding="utf-8",
    )

    matrix, names = load_covmat(covmat_path)

    assert names == ("omega_m", "sigma8", "h")
    np.testing.assert_allclose(
        matrix,
        np.array(
            [
                [4.0, 1.0, 0.2],
                [1.0, 3.0, 0.5],
                [0.2, 0.5, 2.0],
            ]
        ),
    )


def test_compare_covmats_matches_by_header_names(tmp_path: Path) -> None:
    covmat_a = tmp_path / "a.covmat"
    covmat_b = tmp_path / "b.covmat"

    covmat_a.write_text(
        "# omega_m sigma8 h\n"
        "4.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 9.0\n",
        encoding="utf-8",
    )
    covmat_b.write_text(
        "# h sigma8 omega_m\n"
        "9.0 0.0 0.0\n"
        "0.0 0.5 0.0\n"
        "0.0 0.0 8.0\n",
        encoding="utf-8",
    )

    result = compare_covmats(
        covmat_a,
        covmat_b,
        params_A=["omega_m", "sigma8"],
        params_B=["omega_m", "sigma8"],
    )

    np.testing.assert_allclose(result.degradation_factors, np.array([2.0, 0.5]))
    assert result.parameter_names == ("omega_m", "sigma8")


def test_compare_MCMC_chains_uses_getdist_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "chain_a"
    root_b = tmp_path / "chain_b"

    samples_a = MCSamples(
        samples=np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [4.0, 2.0],
            ]
        ),
        names=["omega_m", "sigma8"],
        labels=["omega_m", "sigma8"],
    )
    samples_b = MCSamples(
        samples=np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [8.0, 1.0],
            ]
        ),
        names=["omega_m", "sigma8"],
        labels=["omega_m", "sigma8"],
    )
    samples_a.saveAsText(str(root_a))
    samples_b.saveAsText(str(root_b))

    result = compare_MCMC_chains(
        root_a,
        root_b,
        params_A=["omega_m", "sigma8"],
        params_B=["omega_m", "sigma8"],
        chain_settings={"ignore_rows": 0.0},
    )

    assert result.parameter_names == ("omega_m", "sigma8")
    assert result.degradation_factors.shape == (2,)


def test_eigenmodes_accepts_covmat_only_inputs(tmp_path: Path) -> None:
    covmat_a = tmp_path / "a.covmat"
    covmat_b = tmp_path / "b.covmat"

    covmat_a.write_text(
        "# omega_m sigma8\n"
        "4.0 0.0\n"
        "0.0 1.0\n",
        encoding="utf-8",
    )
    covmat_b.write_text(
        "# omega_m sigma8\n"
        "8.0 0.0\n"
        "0.0 0.5\n",
        encoding="utf-8",
    )

    result = eigenmodes(
        covmat_a=covmat_a,
        covmat_b=covmat_b,
        params_A=["omega_m", "sigma8"],
        params_B=["omega_m", "sigma8"],
    )

    np.testing.assert_allclose(result.degradation_factors, np.array([2.0, 0.5]))
    assert result.reference_mean is None
    assert result.alternative_mean is None


def test_eigenmodes_uses_covmats_and_chain_means_when_both_are_available(
    tmp_path: Path,
) -> None:
    covmat_a = tmp_path / "a.covmat"
    covmat_b = tmp_path / "b.covmat"
    root_a = tmp_path / "chain_a"
    root_b = tmp_path / "chain_b"

    covmat_a.write_text(
        "# omega_m sigma8\n"
        "4.0 0.0\n"
        "0.0 1.0\n",
        encoding="utf-8",
    )
    covmat_b.write_text(
        "# omega_m sigma8\n"
        "8.0 0.0\n"
        "0.0 0.5\n",
        encoding="utf-8",
    )

    samples_a = MCSamples(
        samples=np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 2.0],
            ]
        ),
        names=["omega_m", "sigma8"],
        labels=["omega_m", "sigma8"],
    )
    samples_b = MCSamples(
        samples=np.array(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [1.0, 2.0],
                [3.0, 2.0],
            ]
        ),
        names=["omega_m", "sigma8"],
        labels=["omega_m", "sigma8"],
    )
    samples_a.saveAsText(str(root_a))
    samples_b.saveAsText(str(root_b))

    result = eigenmodes(
        chain_root_a=root_a,
        chain_root_b=root_b,
        covmat_a=covmat_a,
        covmat_b=covmat_b,
        params_A=["omega_m", "sigma8"],
        params_B=["omega_m", "sigma8"],
        chain_settings={"ignore_rows": 0.0},
    )

    np.testing.assert_allclose(result.degradation_factors, np.array([2.0, 0.5]))
    np.testing.assert_allclose(result.reference_mean, np.array([1.0, 1.0]))
    np.testing.assert_allclose(result.alternative_mean, np.array([2.0, 1.0]))


def test_compare_covmats_allows_different_parameter_names(tmp_path: Path) -> None:
    covmat_a = tmp_path / "a_alias.covmat"
    covmat_b = tmp_path / "b_alias.covmat"

    covmat_a.write_text(
        "# theta_s_100 omega_b omega_cdm\n"
        "4.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 9.0\n",
        encoding="utf-8",
    )
    covmat_b.write_text(
        "# theta_s_1e2 omega_b omega_cdm\n"
        "8.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 9.0\n",
        encoding="utf-8",
    )

    result = compare_covmats(
        covmat_a,
        covmat_b,
        params_A=["theta_s_100", "omega_b"],
        params_B=["theta_s_1e2", "omega_b"],
    )

    assert result.parameter_names == ("theta_s_100", "omega_b")
    assert result.alternative_parameter_names == ("theta_s_1e2", "omega_b")
    np.testing.assert_allclose(result.degradation_factors, np.array([2.0, 1.0]))


def test_compare_samples_handles_getdist_names_not_valid_as_attributes() -> None:
    class FakeParamNames:
        def list(self):
            return ["omega_b", "chi2__bao.desi_dr2"]

    class FakeSamples:
        def __init__(self):
            self.samples = np.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                ]
            )
            self.weights = np.ones(3)

        def getParamNames(self):
            return FakeParamNames()

    result = compare_samples(
        FakeSamples(),
        FakeSamples(),
        selected_parameters=["omega_b"],
        selected_parameters_b=["omega_b"],
    )

    assert result.parameter_names == ("omega_b",)


def test_summarize_modes_reports_leading_parameters() -> None:
    result = analyze_covariances(
        np.array([[4.0, 0.0], [0.0, 1.0]]),
        np.array([[8.0, 0.0], [0.0, 0.5]]),
        ["omega_m", "sigma8"],
    )

    summaries = summarize_modes(result, top_n_params=1)

    assert summaries[0].mode_index == 1
    assert summaries[0].status == "degraded"
    assert summaries[0].degradation_factor == 2.0
    assert summaries[0].sigma_ratio == np.sqrt(2.0)
    assert summaries[0].top_parameters[0].reference_name == "omega_m"


def test_top_correlation_changes_orders_by_absolute_delta() -> None:
    result = compare_covariances(
        np.array(
            [
                [1.0, 0.1, 0.0],
                [0.1, 1.0, 0.2],
                [0.0, 0.2, 1.0],
            ]
        ),
        np.array(
            [
                [1.0, 0.8, 0.0],
                [0.8, 1.0, -0.4],
                [0.0, -0.4, 1.0],
            ]
        ),
        ["p1", "p2", "p3"],
    )

    changes = top_correlation_changes(result, top_n=2)

    assert len(changes) == 2
    assert abs(changes[0].delta_correlation) >= abs(changes[1].delta_correlation)
    assert changes[0].parameter_pair_reference == ("p1", "p2")


def test_top_eigenvector_alignments_reports_overlap_structure() -> None:
    result = compare_covariances(
        np.array([[4.0, 0.0], [0.0, 1.0]]),
        np.array([[8.0, 0.0], [0.0, 0.5]]),
        ["omega_m", "sigma8"],
        rotation_basis="original",
    )

    alignments = top_eigenvector_alignments(result, top_n=2)

    assert len(alignments) == 2
    assert alignments[0].reference_mode_index == 1
    assert alignments[0].alternative_mode_index == 1
    assert np.isclose(alignments[0].overlap, 1.0)


def test_format_mode_direction_and_summary_are_readable() -> None:
    result = compare_covariances(
        np.array([[4.0, 0.0], [0.0, 1.0]]),
        np.array([[8.0, 0.0], [0.0, 0.5]]),
        ["omega_m", "sigma8"],
    )

    direction = format_mode_direction(result, mode_index=1, max_terms=2, precision=2)
    summary = format_mode_summary(result, mode_index=1, max_terms=2, precision=2)

    assert "omega_m" in direction
    assert "Mode 1" in summary
    assert "rho =" in summary
    assert "direction:" in summary
    assert "fractional contributions:" not in summary
    assert "direction:" in summary


def test_format_mode_direction_and_summary_can_render_all_modes() -> None:
    result = compare_covariances(
        np.array([[4.0, 0.0], [0.0, 1.0]]),
        np.array([[8.0, 0.0], [0.0, 0.5]]),
        ["omega_m", "sigma8"],
    )

    all_directions = format_mode_direction(result)
    all_summaries = format_mode_summary(result)

    assert "Mode 1:" in all_directions
    assert "Mode 2:" in all_directions
    assert "Mode 1" in all_summaries
    assert "Mode 2" in all_summaries


def test_eigenmode_report_has_systematic_sections() -> None:
    result = compare_covariances(
        np.array([[4.0, 0.0], [0.0, 1.0]]),
        np.array([[8.0, 0.0], [0.0, 0.5]]),
        ["omega_m", "sigma8"],
    )

    report = eigenmode_report(result, max_terms=2, precision=2)

    assert "\nA\n" in "\n" + report
    assert "\nB\n" in "\n" + report
    assert "Shifts" in report
    assert "Rotations" in report
    assert "C = C_A^(-1/2) C_B C_A^(-1/2)" in report
    assert "constructed comparison matrix C:" in report
    assert "Generalized mode 1" in report


def test_eigenmode_report_includes_shift_metrics_when_means_are_available() -> None:
    samples_a = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ]
    )
    samples_b = samples_a + np.array([1.0, 0.0])

    result = compare_samples(
        samples_a,
        samples_b,
        parameter_names=["p1", "p2"],
        parameter_names_b=["p1", "p2"],
    )

    report = eigenmode_report(result, precision=3)

    assert "B shifted from A in A units" in report
    assert "A shifted from B in B units" in report
    assert "joint shift in combined units" in report


def test_eigenmode_report_notes_near_degenerate_rotation_modes() -> None:
    result = compare_covariances(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[2.0, 0.0], [0.0, 0.5]]),
        ["p1", "p2"],
    )

    report = eigenmode_report(result, precision=3, interpretation=True)

    assert "note = A modes 1 and 2 are nearly degenerate" in report


def test_format_comparison_report_remains_as_alias() -> None:
    result = compare_covariances(
        np.array([[4.0, 0.0], [0.0, 1.0]]),
        np.array([[8.0, 0.0], [0.0, 0.5]]),
        ["omega_m", "sigma8"],
    )

    assert format_comparison_report(result) == eigenmode_report(result)


def test_project_samples_onto_modes_recovers_expected_variances() -> None:
    comparison = compare_covariances(
        np.array([[4.0, 0.0], [0.0, 1.0]]),
        np.array([[8.0, 0.0], [0.0, 0.5]]),
        ["omega_m", "sigma8"],
    )

    reference_samples = np.array(
        [
            [2.0, 1.0],
            [2.0, -1.0],
            [-2.0, 1.0],
            [-2.0, -1.0],
        ]
    )
    alternative_samples = np.array(
        [
            [np.sqrt(8.0), np.sqrt(0.5)],
            [np.sqrt(8.0), -np.sqrt(0.5)],
            [-np.sqrt(8.0), np.sqrt(0.5)],
            [-np.sqrt(8.0), -np.sqrt(0.5)],
        ]
    )

    projected_ref = project_samples_onto_modes(
        reference_samples,
        comparison,
        parameter_names=["omega_m", "sigma8"],
        use_normalized_modes=False,
    )
    projected_alt = project_samples_onto_modes(
        alternative_samples,
        comparison,
        dataset="alternative",
        parameter_names=["omega_m", "sigma8"],
        use_normalized_modes=False,
        reference_center=np.zeros(2),
    )

    np.testing.assert_allclose(np.var(projected_ref, axis=0), np.array([1.0, 1.0]))
    np.testing.assert_allclose(
        np.var(projected_alt, axis=0),
        comparison.degradation_factors,
    )


def test_get_mode_samples_returns_both_projected_arrays() -> None:
    comparison = compare_covariances(
        np.array([[4.0, 0.0], [0.0, 1.0]]),
        np.array([[8.0, 0.0], [0.0, 0.5]]),
        ["omega_m", "sigma8"],
    )

    reference_samples = np.array(
        [
            [2.0, 1.0],
            [2.0, -1.0],
            [-2.0, 1.0],
            [-2.0, -1.0],
        ]
    )
    alternative_samples = np.array(
        [
            [np.sqrt(8.0), np.sqrt(0.5)],
            [np.sqrt(8.0), -np.sqrt(0.5)],
            [-np.sqrt(8.0), np.sqrt(0.5)],
            [-np.sqrt(8.0), -np.sqrt(0.5)],
        ]
    )

    projected_ref, projected_alt = get_mode_samples(
        reference_samples,
        alternative_samples,
        comparison,
        parameter_names_a=["omega_m", "sigma8"],
        parameter_names_b=["omega_m", "sigma8"],
    )

    assert projected_ref.shape == (4, 2)
    assert projected_alt.shape == (4, 2)


def test_get_mode_samples_can_use_raw_or_normalized_modes() -> None:
    comparison = compare_covariances(
        np.array([[4.0, 0.0], [0.0, 1.0]]),
        np.array([[8.0, 0.0], [0.0, 0.5]]),
        ["omega_m", "sigma8"],
    )

    reference_samples = np.array(
        [
            [2.0, 1.0],
            [2.0, -1.0],
            [-2.0, 1.0],
            [-2.0, -1.0],
        ]
    )
    alternative_samples = np.array(
        [
            [np.sqrt(8.0), np.sqrt(0.5)],
            [np.sqrt(8.0), -np.sqrt(0.5)],
            [-np.sqrt(8.0), np.sqrt(0.5)],
            [-np.sqrt(8.0), -np.sqrt(0.5)],
        ]
    )

    norm_ref, norm_alt = get_mode_samples(
        reference_samples,
        alternative_samples,
        comparison,
        parameter_names_a=["omega_m", "sigma8"],
        parameter_names_b=["omega_m", "sigma8"],
        use_normalized_modes=True,
    )
    raw_ref, raw_alt = get_mode_samples(
        reference_samples,
        alternative_samples,
        comparison,
        parameter_names_a=["omega_m", "sigma8"],
        parameter_names_b=["omega_m", "sigma8"],
        use_normalized_modes=False,
    )

    assert np.all(np.isfinite(norm_ref))
    assert np.all(np.isfinite(norm_alt))
    assert np.all(np.isfinite(raw_ref))
    assert np.all(np.isfinite(raw_alt))


def test_add_mode_derived_parameter_adds_getdist_field() -> None:
    samples = MCSamples(
        samples=np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        ),
        names=["p1", "p2"],
        labels=["p1", "p2"],
    )
    comparison = compare_covariances(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[2.0, 0.0], [0.0, 0.5]]),
        ["p1", "p2"],
    )

    derived = add_mode_derived_parameter(samples, comparison, mode_index=1)

    assert "mode_1" in derived.getParamNames().list()


def test_add_all_mode_derived_parameters_adds_every_mode() -> None:
    samples = MCSamples(
        samples=np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        ),
        names=["p1", "p2"],
        labels=["p1", "p2"],
    )
    comparison = compare_covariances(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[2.0, 0.0], [0.0, 0.5]]),
        ["p1", "p2"],
    )

    derived = add_all_mode_derived_parameters(samples, comparison)

    assert "mode_1" in derived.getParamNames().list()
    assert "mode_2" in derived.getParamNames().list()
