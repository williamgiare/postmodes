"""Optional sampling-stability checks; these do not calibrate tension significance."""

import warnings

import numpy as np

from .api import _coerce_samples, compare_samples


def bootstrap_stability(
    chains_a, chains_b, *, parameter_names_a=None, parameter_names_b=None,
    selected_parameters=None, selected_parameters_b=None,
    block_length, n_resamples=200, seed=None,
):
    """Circular moving-block bootstrap within each independent chain.

    Pass lists of chronological post-burn-in arrays or single-chain MCSamples.
    Row weights travel with the rows. Block length is measured in stored rows,
    and must be chosen after checking autocorrelation and sensitivity to length.
    A and B are independently resampled: this estimates Monte Carlo stability,
    not a sampling distribution of tension from shared observational data.
    Sorted rho values are ranked spectra, not matched eigenvector identities.
    """
    if not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer.")
    if not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")
    def prepare(chains, names):
        if not isinstance(chains, (list, tuple)) or not chains:
            raise ValueError("Pass a non-empty list of independent, chronological chains.")
        prepared = []
        common = None
        for chain in chains:
            matrix, current_names, weights = _coerce_samples(chain, parameter_names=names)
            offsets = getattr(chain, "chain_offsets", None)
            if offsets is not None and len(offsets) > 2:
                raise ValueError("Split merged GetDist samples with getSeparateChains() before bootstrapping.")
            if len(matrix) < 2 * block_length:
                raise ValueError("Each chain must contain at least two blocks.")
            if common is not None and current_names != common:
                raise ValueError("Parameter order must agree within each chain group.")
            common = current_names
            prepared.append((matrix, np.ones(len(matrix)) if weights is None else weights))
        return prepared, common
    a, names_a = prepare(chains_a, parameter_names_a)
    b, names_b = prepare(chains_b, parameter_names_a if parameter_names_b is None else parameter_names_b)
    def merge(group, rng=None):
        matrices, weights = [], []
        for matrix, weight in group:
            if rng is None:
                idx = np.arange(len(matrix))
            else:
                starts = rng.integers(len(matrix), size=int(np.ceil(len(matrix)/block_length)))
                idx = ((starts[:, None]+np.arange(block_length)) % len(matrix)).ravel()[:len(matrix)]
            matrices.append(matrix[idx]); weights.append(weight[idx])
        return np.concatenate(matrices), np.concatenate(weights)
    def calculate(rng=None):
        x, wx = merge(a, rng); y, wy = merge(b, rng)
        r = compare_samples(x, y, parameter_names=names_a, parameter_names_b=names_b,
                            weights_a=wx, weights_b=wy, selected_parameters=selected_parameters,
                            selected_parameters_b=selected_parameters_b)
        return np.array([*r.shifts.values(), r.alpha, r.A_aniso, *r.degradation_factors])
    estimate = calculate()
    rng = np.random.default_rng(seed)
    draws = np.full((n_resamples, len(estimate)), np.nan)
    for i in range(n_resamples):
        try:
            draws[i] = calculate(rng)
        except (ValueError, np.linalg.LinAlgError):
            continue
    failures = int(np.isnan(draws).any(axis=1).sum())
    if failures == n_resamples:
        raise ValueError("All bootstrap replicates failed; inspect chains and block length.")
    if failures:
        warnings.warn(f"{failures}/{n_resamples} bootstrap replicates failed; intervals exclude them and may be unreliable.", UserWarning)
    labels = ['B_in_A', 'A_in_B', 'SCCD', 'alpha', 'A_aniso'] + [f'rho_{i+1}' for i in range(len(estimate)-5)]
    return dict(names=labels, estimate=estimate, replicates=draws,
                quantile_levels=np.array([.16, .5, .84]), quantiles=np.nanquantile(draws, [.16, .5, .84], axis=0),
                failed_resamples=failures, block_length=block_length)
