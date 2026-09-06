# API

This page documents the public user-facing API.

## Main Entry Point

### `eigenmodes(...)`

This is the main function for posterior comparison.

Minimal usage: `result = eigenmodes("chains/A", "chains/B")`.
Pass `params=[...]` to choose a common list, or `params_A` and `params_B` to map
different names. With no lists, common names are selected; see [Input](input.md).
`rotation_basis="reference_standardized"` is the default; `"original"` retains
the previous rotation metric. `check_covariances=True` adds an optional check
when both covmat files and chain roots are supplied.

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
from postmodes import eigenmodes

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
from postmodes import eigenmode_report

print(eigenmode_report(result, precision=4))
```

Reports are numerical by default. Add `interpretation=True` for concise notes.
`format_mode_summary` accepts the same flag. `result.shifts` returns a dictionary
with `B_in_A`, `A_in_B`, `SCCD`, or None when no means are available.
`result.alpha` and `result.A_aniso` expose the deformation scalars directly.

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

The bases are those specified by `result.rotation_basis`, standardized relative
to A by default. Raw overlaps remain at `result.original_eigenvector_overlap`.
`rotation_subspace_angles(result, [1, 2], [1, 2])` returns principal angles in
degrees for selected one-based mode clusters. See [Mathematics](mathematics.md).

### `top_correlation_changes(result, ...)`

Returns the largest pairwise correlation changes between `A` and `B`.

## Sample Projection Utilities

### `get_mode_samples(...)`

Projects samples onto the generalized comparison modes and returns arrays of mode coefficients for both posterior samples.

Defaults: `normalization="reference"`, `center="reference"`. Both chains use
the same origin and coefficients, preserving shifts. `normalization="euclidean"`
and `center="separate"` recover the old defaults. The legacy boolean
`use_normalized_modes` overrides normalization; `center=True/False` maps to
`"separate"/"none"`. `weights_a` and `weights_b` support weighted raw arrays.
The returned arrays do not encode weights: pass chain weights to histograms/KDEs.

### `add_all_mode_derived_parameters(...)`

Adds generalized modes to a GetDist `MCSamples` object as derived parameters.

This is useful for `triangle_plot` and `plots_2d` in mode space.

These helpers accept `normalization`, `center`, and `reference_center`, preserve
weights, and return copies. With a covariance-only result, supply the same
reference mean to both calls using `reference_center=...`. The paired
`get_mode_samples` helper can infer that mean from A automatically.

## Sampling Stability

`bootstrap_stability(chains_a, chains_b, block_length=..., n_resamples=200,
seed=..., parameter_names_a=..., parameter_names_b=...)` accepts lists of
individual chronological post-burn-in chains. It returns estimates, replicates,
16/50/84-percentiles and failure counts for shifts, alpha, A_aniso and sorted rho.
The user chooses block length in stored rows; this is a Monte Carlo stability
diagnostic, not a significance calibration. See [Validation](validation.md).

## Terminal

```bash
postmodes chains/A chains/B --params omega_b omega_cdm
postmodes chains/A chains/B --interpretation
postmodes --covmat-a A.covmat --covmat-b B.covmat --json
python -m postmodes --help
```

GetDist progress goes to stderr, report/JSON to stdout. Use `--ignore-rows 0.3`,
`--params-a ... --params-b ...` for mappings, and `--check-covariances` to check
combined inputs. Install with `pip install -e .` to register the console command.

## Lower-Level APIs

These remain available for advanced use, but `eigenmodes(...)` should be preferred for most workflows:

- `compare_covariances(...)`
- `compare_samples(...)`
- `compare_covmats(...)`
- `compare_MCMC_chains(...)`
- `analyze_covariance(...)`
- `analyze_covariances(...)`
