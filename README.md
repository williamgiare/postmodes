# Posterior Eigenmodes

`posterior-eigenmodes` is a Python package for comparing posterior constraints through the geometry of their covariance matrices.

The package is built around one scientific question:

**given two posterior distributions on the same physical parameter space, which parameter combinations are preserved, which are degraded or tightened, and how does the posterior geometry change?**

It is especially aimed at covariance- and chain-based analyses in cosmology, with direct support for MCMC chains and covariance matrices.

## What The Package Does

`posterior-eigenmodes` compares a reference posterior `A` and an alternative posterior `B` by combining:

- PCA of `A`
- PCA of `B`
- mean-shift diagnostics
- rotation diagnostics between the PCA bases
- generalized eigenmode analysis of

```math
C = C_A^{-1/2} C_B C_A^{-1/2}
```

From this it extracts:

- mode-by-mode variance ratios `rho`
- standard-deviation ratios `sqrt(rho)`
- generalized parameter combinations in the original basis
- isotropic and anisotropic deformation metrics
- correlation-structure changes between posteriors

## Installation

From the repository root:

```bash
pip install -e .
```

## Example

Two chain roots are enough (30% burn-in by default):

```python
from posterior_eigenmodes import eigenmodes, eigenmode_report

result = eigenmodes("chains/A", "chains/B")
print(eigenmode_report(result))
# Add interpretation=True for concise explanatory notes.
```

Automatic selection uses exact common non-derived names, including any shared
nuisance parameters. Check the selected list printed in the report. To select
physical parameters explicitly or map different names:

```python
from posterior_eigenmodes import eigenmodes, eigenmode_report

result = eigenmodes(
    covmat_a="path/to/A.covmat",
    covmat_b="path/to/B.covmat",
    chain_root_a="path/to/chains/A",
    chain_root_b="path/to/chains/B",
    params_A=["omega_b", "omega_cdm", "theta_s_1e2", "delta_DMDE"],
    params_B=["omega_b", "omega_cdm", "theta_s_100", "delta_DMDE"],
)

print(eigenmode_report(result, precision=4))
```

From the terminal, after installation:

```bash
posterior-eigenmodes chains/A chains/B --params omega_b omega_cdm
posterior-eigenmodes chains/A chains/B --interpretation
python -m posterior_eigenmodes --help
```

Version 1.0.0 uses rotations in common reference-sigma coordinates and weighted
mode plots with a common origin. Scalar comparison indicators retain their
definitions. See [migration notes](docs/migration-1.0.md) for changed defaults.

It supports three usage patterns:

1. covariance matrices only
2. chains only
3. covariance matrices + chains together

## Main Outputs

- PCA eigenvalues and eigenvectors of `A`
- PCA eigenvalues and eigenvectors of `B`
- mean-shift metrics between `A` and `B`
- overlap matrix between the PCA bases
- generalized eigenvalues `rho`
- generalized modes in the original parameter basis
- isotropic deformation `alpha`
- anisotropic deformation `A_aniso`
- correlation matrices and largest correlation changes

## Project Structure

- [posterior_eigenmodes/api.py](posterior_eigenmodes/api.py)
  high-level entry points such as `eigenmodes(...)`
- [posterior_eigenmodes/io.py](posterior_eigenmodes/io.py)
  covariance and chain loading
- [posterior_eigenmodes/stats.py](posterior_eigenmodes/stats.py)
  weighted means and covariances from samples
- [posterior_eigenmodes/modes.py](posterior_eigenmodes/modes.py)
  PCA and generalized eigenmode analysis
- [posterior_eigenmodes/interpretation.py](posterior_eigenmodes/interpretation.py)
  readable summaries and report generation
- [posterior_eigenmodes/plotting.py](posterior_eigenmodes/plotting.py)
  mode projections and plotting helpers

## Documentation

- [docs/README.md](docs/README.md)
- [docs/mathematics.md](docs/mathematics.md)
- [docs/api.md](docs/api.md)
- [docs/input.md](docs/input.md)
- [docs/output.md](docs/output.md)
- [docs/validation.md](docs/validation.md)
- [docs/notebooks.md](docs/notebooks.md)

## Notebooks

- [1_Theory.ipynb](notebooks/1_Theory.ipynb)
  theory-first explanation of the geometry
- [2_Validation.ipynb](notebooks/2_Validation.ipynb)
  synthetic validation cases
- [3_Basic_Example.ipynb](notebooks/3_Basic_Example.ipynb)
  basic real-data workflow example
- [4_Advanced_Example.ipynb](notebooks/4_Advanced_Example.ipynb)
  advanced real-data comparison example
