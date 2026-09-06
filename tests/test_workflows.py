import json
import subprocess
import sys

import numpy as np
import pytest

from postmodes import eigenmodes, eigenmode_report, bootstrap_stability, analyze_covariances, rotation_subspace_angles
from postmodes import api


def write_chain(root, names, values, weights):
    np.savetxt(str(root)+'.1.txt', np.column_stack([weights, np.zeros(len(values)), values]), header='weight minuslogpost '+' '.join(names))
    root.with_suffix('.paramnames').write_text('\n'.join(f'{n} {n}' for n in names))


def test_chain_auto_selection_and_single_load(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    for name in ['A', 'B']:
        write_chain(tmp_path/name, ['x', 'y'], rng.normal(size=(100, 2)), np.ones(100))
    loader = api.load_getdist_chain
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[0])
        return loader(*args, **kwargs)
    monkeypatch.setattr(api, 'load_getdist_chain', counted)
    r = eigenmodes(tmp_path/'A', tmp_path/'B')
    assert len(calls) == 2
    assert r.parameter_names == ('x', 'y')
    assert 'interpretation =' not in eigenmode_report(r)
    assert 'interpretation =' in eigenmode_report(r, interpretation=True)


def test_default_burnin_and_retained_row_weights(tmp_path):
    from postmodes.stats import weighted_mean, weighted_covariance
    rng = np.random.default_rng(23)
    arrays = [rng.normal(size=(100, 2)), rng.normal(size=(100, 2)) + 2]
    weights = np.arange(1., 101.)
    for name, x in zip(['A', 'B'], arrays):
        write_chain(tmp_path/name, ['x', 'y'], x, weights)
    result = eigenmodes(tmp_path/'A', tmp_path/'B', params=np.array(['x', 'y']))
    np.testing.assert_allclose(result.reference_mean, weighted_mean(arrays[0][30:], weights[30:]))
    np.testing.assert_allclose(result.alternative_mean, weighted_mean(arrays[1][30:], weights[30:]))
    np.testing.assert_allclose(result.reference_covariance, weighted_covariance(arrays[0][30:], weights[30:]))
    proc = subprocess.run([sys.executable, '-m', 'postmodes', str(tmp_path/'A'), str(tmp_path/'B'), '--json'], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    np.testing.assert_allclose(json.loads(proc.stdout)['reference_mean'], result.reference_mean)


def test_covmat_consistency_and_cli(tmp_path):
    rng = np.random.default_rng(14)
    x = rng.normal(size=(120, 2))
    for name in ['A', 'B']:
        write_chain(tmp_path/name, ['x', 'y'], x, np.ones(len(x)))
        np.savetxt(tmp_path/f'{name}.covmat', np.eye(2)*20, header='x y')
    with pytest.warns(UserWarning, match='supplied covariance differs'):
        r = eigenmodes(tmp_path/'A', tmp_path/'B', covmat_a=tmp_path/'A.covmat', covmat_b=tmp_path/'B.covmat', check_covariances=True)
    np.testing.assert_array_equal(r.reference_covariance, np.eye(2)*20)
    assert len(r.input_covariance_diagnostics) == 2
    proc = subprocess.run([sys.executable, '-m', 'postmodes', '--covmat-a', str(tmp_path/'A.covmat'), '--covmat-b', str(tmp_path/'B.covmat'), '--json'], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data['alpha'] == pytest.approx(1)
    assert data['shifts'] is None


def test_bootstrap_reproducible_and_subspace_invariant():
    rng = np.random.default_rng(6)
    a = [rng.normal(size=(180, 2)), rng.normal(size=(150, 2))]
    b = [rng.normal(size=(200, 2))*[2, .5]+[1, 0]]
    kw = dict(parameter_names_a=['x', 'y'], block_length=10, n_resamples=15, seed=7)
    r = bootstrap_stability(a, b, **kw)
    t = bootstrap_stability(a, b, **kw)
    np.testing.assert_array_equal(r['replicates'], t['replicates'])
    assert r['failed_resamples'] == 0
    assert np.all(np.isfinite(r['quantiles']))
    assert np.all(np.std(r['replicates'], axis=0) > 0)
    with pytest.raises(ValueError, match='two blocks'):
        bootstrap_stability(a, b, **(kw | dict(block_length=100)))
    C = np.diag([4., 4., 1.])
    r = analyze_covariances(C, C, ['x', 'y', 'z'], rotation_basis='original')
    np.testing.assert_allclose(rotation_subspace_angles(r, [1, 2], [1, 2]), 0, atol=1e-12)


def test_bootstrap_getdist_separate_chains_keep_weights():
    from getdist import MCSamples
    from postmodes.api import compare_samples
    rng = np.random.default_rng(32)
    x = [rng.normal(size=(100, 2)), rng.normal(size=(100, 2))]
    weights = [rng.integers(1, 30, 100), rng.integers(1, 30, 100)]
    a = MCSamples(samples=x, weights=weights, loglikes=[np.zeros(100)]*2, names=['x', 'y'])
    b = MCSamples(samples=[c * [2, .5] + 1 for c in x], weights=weights, loglikes=[np.zeros(100)]*2, names=['x', 'y'])
    kw = dict(block_length=10, n_resamples=5, seed=42)
    with pytest.raises(ValueError, match='Split merged'):
        bootstrap_stability([a], [b], **kw)
    result = bootstrap_stability(a.getSeparateChains(), b.getSeparateChains(), parameter_names_a=a.getParamNames().list(), parameter_names_b=b.getParamNames().list(), **kw)
    direct = compare_samples(a, b)
    np.testing.assert_allclose(result['estimate'], [*direct.shifts.values(), direct.alpha, direct.A_aniso, *direct.degradation_factors])
    assert result['failed_resamples'] == 0
