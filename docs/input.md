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

## Burn-In and Weights

When chains are used:

- burn-in is handled through GetDist loading settings,
- sample weights are used when computing weighted means and weighted covariances.

So posterior means and covariances are estimated from the post-burn-in weighted samples.
