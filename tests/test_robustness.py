import numpy as np
import pytest
from scipy.linalg import eigh
from getdist import MCSamples

from posterior_eigenmodes import analyze_covariances, get_mode_samples, eigenmode_report
from posterior_eigenmodes.api import compare_samples
from posterior_eigenmodes.stats import weighted_covariance, weighted_mean
from posterior_eigenmodes.plotting import plot_mode_distributions, add_all_mode_derived_parameters


def matrices(seed=4):
    rng = np.random.default_rng(seed)
    X, Y = rng.normal(size=(2, 4, 4))
    return X @ X.T + np.eye(4), Y @ Y.T + np.eye(4)


@pytest.mark.parametrize('seed', range(8))
def test_solver_identities_and_symmetric_C(seed):
    A, B = matrices(seed)
    r = analyze_covariances(A, B, list('abcd'))
    V = r.mode_vectors
    np.testing.assert_allclose(r.degradation_factors, eigh(B, A, eigvals_only=True)[::-1], rtol=1e-11)
    np.testing.assert_allclose(V.T @ A @ V, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(V.T @ B @ V, np.diag(r.degradation_factors), atol=1e-12)
    vals, vecs = np.linalg.eigh(A)
    invsqrt = (vecs / np.sqrt(vals)) @ vecs.T
    np.testing.assert_allclose(r.whitened_comparison_matrix, invsqrt @ B @ invsqrt, atol=1e-12)
    np.testing.assert_allclose(r.mode_vectors, invsqrt @ r.whitened_eigenvectors, atol=1e-12)


def test_unit_invariance_and_swap():
    A, B = matrices()
    mu = np.array([1., 2., -1., .4])
    r = analyze_covariances(A, B, list('abcd'), reference_mean=np.zeros(4), alternative_mean=mu)
    D = np.diag([1e-10, 1e8, 1e-4, 1e3])
    t = analyze_covariances(D @ A @ D, D @ B @ D, list('abcd'), reference_mean=np.zeros(4), alternative_mean=D @ mu)
    np.testing.assert_allclose(r.eigenvector_overlap, t.eigenvector_overlap, atol=1e-11)
    assert np.all(t.reference_eigenvalues > 0)
    assert np.all(t.alternative_eigenvalues > 0)
    assert np.log(t.reference_eigenvalues).sum() == pytest.approx(np.linalg.slogdet(D @ A @ D)[1], abs=1e-7)
    np.testing.assert_allclose(r.degradation_factors, t.degradation_factors, rtol=1e-11)
    np.testing.assert_allclose([r.alpha, r.A_aniso, *r.shifts.values()], [t.alpha, t.A_aniso, *t.shifts.values()], rtol=1e-11)
    reverse = analyze_covariances(B, A, list('abcd'), reference_mean=mu, alternative_mean=np.zeros(4))
    np.testing.assert_allclose(reverse.degradation_factors, 1 / r.degradation_factors[::-1])
    assert reverse.alpha == pytest.approx(1/r.alpha)
    assert reverse.A_aniso == pytest.approx(r.A_aniso)
    assert reverse.shifts['SCCD'] == pytest.approx(r.shifts['SCCD'])


def test_full_mean_reordering_and_explicit_selected_means():
    A = np.diag([1., 100.])
    kwargs = dict(selected_parameters=['y', 'x'], reference_mean=np.zeros(2))
    r = analyze_covariances(A, A, ['x', 'y'], alternative_mean=np.array([1., 0.]), **kwargs)
    np.testing.assert_array_equal(r.alternative_mean, [0., 1.])
    assert r.shifts['B_in_A'] == pytest.approx(1.)
    t = analyze_covariances(A, A, ['x', 'y'], alternative_mean=np.array([0., 1.]), mean_order='selected', **kwargs)
    np.testing.assert_array_equal(t.alternative_mean, r.alternative_mean)


def test_single_pca_uses_relative_not_absolute_floor():
    from posterior_eigenmodes import analyze_covariance
    A = np.diag([1e-24, 4e-24])
    result = analyze_covariance(A, ['x', 'y'], eigenvalue_floor=1e-14)
    np.testing.assert_allclose(result.eigenvalues, [4e-24, 1e-24], rtol=1e-12, atol=0)


def test_affine_invariance_of_comparison_indicators():
    A, B = matrices()
    T = np.array([[1., 2., 0., 0.], [0., 1., 1., 0.], [0., 0., 2., 1.], [.1, 0., 0., 1.]])
    mu = np.arange(4.)
    r = analyze_covariances(A, B, list('abcd'), reference_mean=np.zeros(4), alternative_mean=mu)
    transformed = analyze_covariances(T@A@T.T, T@B@T.T, list('abcd'), reference_mean=np.zeros(4), alternative_mean=T@mu)
    np.testing.assert_allclose(r.degradation_factors, transformed.degradation_factors, rtol=1e-11)
    np.testing.assert_allclose([r.alpha, r.A_aniso, *r.shifts.values()], [transformed.alpha, transformed.A_aniso, *transformed.shifts.values()], rtol=1e-11)


@pytest.mark.parametrize('bad', [np.ones((2, 2)), np.array([[1., 2.], [2., 1.]])])
def test_invalid_covariances_rejected_in_both_positions(bad):
    for A, B in [(bad, np.eye(2)), (np.eye(2), bad)]:
        with pytest.raises(ValueError):
            analyze_covariances(A, B, ['x', 'y'])


def test_weighted_projection_preserves_shift_and_both_normalizations():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 2)) * [2., .3]
    w = rng.integers(1, 30, size=400).astype(float)
    a = MCSamples(samples=x, weights=w, names=['x', 'y'])
    b = MCSamples(samples=x * [1.5, .5] + [4, -3], weights=w[::-1], names=['x', 'y'])
    r = compare_samples(a, b)
    for normalization in ['reference', 'euclidean']:
        ma, mb = get_mode_samples(a, b, r, normalization=normalization)
        ca, cb = weighted_covariance(ma, w), weighted_covariance(mb, w[::-1])
        np.testing.assert_allclose(np.diag(cb)/np.diag(ca), r.degradation_factors)
        if normalization == 'reference':
            np.testing.assert_allclose(ca, np.eye(2), atol=1e-12)
            np.testing.assert_allclose(weighted_mean(mb, w[::-1]), (r.alternative_mean-r.reference_mean)@r.mode_vectors)
    da = add_all_mode_derived_parameters(a, r)
    db = add_all_mode_derived_parameters(b, r, dataset='alternative')
    np.testing.assert_array_equal(da.weights, a.weights)
    np.testing.assert_allclose(db.samples[:, -2:], get_mode_samples(a, b, r)[1])
    _, centered_b = get_mode_samples(a, b, r, center='separate')
    np.testing.assert_allclose(weighted_mean(centered_b, b.weights), 0, atol=1e-12)


def test_histogram_uses_weights_and_shared_edges(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt
    x = np.array([[-1., 0.], [0., 2.], [2., -1.], [1., 3.]])
    a = MCSamples(samples=x, weights=np.array([1, 20, 3, 2]), names=['x', 'y'])
    b = MCSamples(samples=x+2, weights=np.array([3, 2, 10, 1]), names=['x', 'y'])
    calls = []
    original = Axes.hist
    def capture(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Axes, 'hist', capture)
    fig, _ = plot_mode_distributions(a, b, compare_samples(a, b))
    np.testing.assert_array_equal(calls[0]['weights'], a.weights)
    np.testing.assert_array_equal(calls[1]['weights'], b.weights)
    np.testing.assert_array_equal(calls[0]['bins'], calls[1]['bins'])
    plt.close(fig)


def test_large_weights_and_degenerate_rotation_note():
    x = np.array([[0., 1.], [2., 3.]])
    np.testing.assert_allclose(weighted_mean(x, [1e308, 1e308]), [1., 2.])
    r = analyze_covariances(np.diag([4., 1.]), np.diag([8., .5]), ['x', 'y'])
    report = eigenmode_report(r, interpretation=True)
    assert 'individual-axis rotation is ambiguous' in report
    assert 'basis = reference_standardized' in report
