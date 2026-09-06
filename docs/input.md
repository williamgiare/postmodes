# Input

## Supported Input Modes

The package supports:

- Cobaya covariance-matrix files,
- Cobaya/GetDist chains,
- or both together.

The main public entry point is:

```python
result = eigenmodes(...)
```

For the simplest chain workflow:

```python
result = eigenmodes("chains/A", "chains/B")
```

Without parameter lists, exact common non-derived GetDist parameter names are
selected in A order. Shared nuisance parameters are included; this is not an
automatic cosmological-parameter classifier. For covariance files, common header
names are used. Unmatched names are excluded with a warning. Always inspect the
parameter list printed at the beginning of the report, especially for a paper.
Use `params=["omega_b", "omega_cdm"]` for a shared explicit selection.

## Parameter Matching

The package uses:

- `params_A`
- `params_B`

to define the common physical parameter subspace.

These lists must have the same length, and position matters:

- `params_A[i]` and `params_B[i]` are assumed to represent the same physical parameter,
- the names do not need to match literally.

Example:

```python
params_A = ["theta_s_1e2", "omega_b", "omega_cdm"]
params_B = ["theta_s_100", "omega_b", "omega_cdm"]
```

This is the intended way to handle different naming conventions between datasets or files.

Matched entries must already have identical physical definitions and units.
Name matching never performs a unit conversion. Duplicate and empty selections
are rejected. Different model dimensions or selected subspaces change the meaning
of the summary statistics and must be stated when comparing models in a table.

## Covariance-Only Mode

```python
result = eigenmodes(
    covmat_a="A.covmat",
    covmat_b="B.covmat",
    params_A=params_A,
    params_B=params_B,
)
```

In this mode:

- the covariance matrices are read from file,
- no mean-shift diagnostics are computed unless means are supplied through a lower-level API.

## Chain-Only Mode

```python
result = eigenmodes(
    chain_root_a="chains/A",
    chain_root_b="chains/B",
    params_A=params_A,
    params_B=params_B,
)
```

In this mode:

- means and covariances are reconstructed from the weighted chain samples,
- burn-in removal follows the GetDist loader settings,
- the `Shifts` section is available.

## Combined Mode

```python
result = eigenmodes(
    covmat_a="A.covmat",
    covmat_b="B.covmat",
    chain_root_a="chains/A",
    chain_root_b="chains/B",
    params_A=params_A,
    params_B=params_B,
)
```

In this mode:

- covariance analysis uses the supplied covariance matrices,
- posterior means are estimated from the chains,
- this avoids reconstructing covariances unnecessarily while still enabling the `Shifts` section.

Set `check_covariances=True` to additionally estimate weighted chain covariances
and compare them to supplied files. The supplied matrices are still used.
`covariance_check_tolerance=0.1` is the default heuristic warning threshold.
The check is optional because reconstructing chain covariances has a cost.
File provenance matters: a sampler proposal covariance is not guaranteed to equal
the posterior covariance for the retained samples.

## Burn-In and Weights

When chains are used:

- burn-in is handled through GetDist loading settings,
- sample weights are used when computing weighted means and weighted covariances.

So posterior means and covariances are estimated from the post-burn-in weighted samples.

The default is `chain_settings={"ignore_rows": 0.3}`. Each root is loaded once,
with GetDist's cache disabled. Already-loaded MCSamples passed to `compare_samples`
are used as-is: apply burn-in before passing them.

For the direct array API, full mean vectors are in the original input parameter
order and are reordered alongside the covariance. Already-selected vectors can
be passed using `mean_order="selected"`; use this explicitly when the selection
is a permutation of all parameters and both possible vector lengths coincide.
