# API

This page documents the public user-facing API.

## Main Entry Point

### `eigenmodes(...)`

This is the main function for posterior comparison.

It supports three input patterns:

1. **Covariance matrices only**
   - `covmat_a`
   - `covmat_b`
   - `params_A`
   - `params_B`

2. **Chains only**
   - `chain_root_a`
   - `chain_root_b`
   - `params_A`
   - `params_B`

3. **Both covariance matrices and chains**
   - covariance matrices are used for the covariance/eigenmode analysis,
   - chains are used only to estimate posterior means for the `Shifts` section.

Typical usage:

```python
from posterior_eigenmodes import eigenmodes

result = eigenmodes(
    covmat_a="path/to/A.covmat",
    covmat_b="path/to/B.covmat",
    chain_root_a="path/to/chain_A",
    chain_root_b="path/to/chain_B",
    params_A=["omega_b", "omega_cdm", "theta_s_1e2", "delta_DMDE"],
    params_B=["omega_b", "omega_cdm", "theta_s_100", "delta_DMDE"],
)
```

## Main Reporting Function

### `eigenmode_report(result, ...)`

Formats a systematic human-readable report with sections:

- `A`
- `B`
- `Shifts`
- `Rotations`
- `C = C_A^(-1/2) C_B C_A^(-1/2)`

Typical usage:

```python
from posterior_eigenmodes import eigenmode_report

print(eigenmode_report(result, precision=4))
```

## Mode Utilities

### `format_mode_direction(result, ...)`

Formats one or all generalized modes as readable linear combinations of parameters.

### `format_mode_summary(result, ...)`

Returns a compact summary of one or all generalized modes, including:

- `rho`,
- `sqrt(rho)`,
- interpretation,
- direction.

### `summarize_modes(result, ...)`

Structured summary of generalized modes for programmatic use.

## Rotation and Correlation Utilities

### `top_eigenvector_alignments(result, ...)`

Returns the largest overlaps between the PCA bases of `A` and `B`.

### `top_correlation_changes(result, ...)`

Returns the largest pairwise correlation changes between `A` and `B`.

## Sample Projection Utilities

### `get_mode_samples(...)`

Projects samples onto the generalized comparison modes and returns arrays of mode coefficients for both posterior samples.

### `add_all_mode_derived_parameters(...)`

Adds generalized modes to a GetDist `MCSamples` object as derived parameters.

This is useful for `triangle_plot` and `plots_2d` in mode space.

## Lower-Level APIs

These remain available for advanced use, but `eigenmodes(...)` should be preferred for most workflows:

- `compare_covariances(...)`
- `compare_samples(...)`
- `compare_covmats(...)`
- `compare_MCMC_chains(...)`
- `analyze_covariance(...)`
- `analyze_covariances(...)`
