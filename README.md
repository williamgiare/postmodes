# PostModes

**Eigenmode analysis of posterior geometry.**

`postmodes` is a Python package for comparing posterior constraints through the geometry of their covariance matrices.

The package is built around one scientific question:

**given two posterior distributions on the same physical parameter space, which parameter combinations are preserved, which are degraded or tightened, and how does the posterior geometry change?**

It is especially aimed at covariance- and chain-based analyses in cosmology, with direct support for MCMC chains and covariance matrices.

## What The Package Does

`postmodes` compares a reference posterior `A` and an alternative posterior `B` by combining:

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
from postmodes import eigenmodes, eigenmode_report

result = eigenmodes("chains/A", "chains/B")
print(eigenmode_report(result))
# Add interpretation=True for concise explanatory notes.
```

Automatic selection uses exact common non-derived names, including any shared
nuisance parameters. Check the selected list printed in the report. To select
physical parameters explicitly or map different names:

```python
from postmodes import eigenmodes, eigenmode_report

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
postmodes chains/A chains/B --params omega_b omega_cdm
postmodes chains/A chains/B --interpretation
python -m postmodes --help
```

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

- [postmodes/api.py](postmodes/api.py)
  high-level entry points such as `eigenmodes(...)`
- [postmodes/io.py](postmodes/io.py)
  covariance and chain loading
- [postmodes/stats.py](postmodes/stats.py)
  weighted means and covariances from samples
- [postmodes/modes.py](postmodes/modes.py)
  PCA and generalized eigenmode analysis
- [postmodes/interpretation.py](postmodes/interpretation.py)
  readable summaries and report generation
- [postmodes/plotting.py](postmodes/plotting.py)
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

## Citation

If you use PostModes in your research, please cite the accompanying paper,

Pedrotti, **Giarè**, Cheng, Di Valentino,
"When, Why and How CMB compression Fails" -- arXiv:2610.nnnnn.

which presents the methodology and its application to the comparison of full and
compressed CMB constraints.

<details>
<summary>BibTeX (provisional)</summary>

```bibtex
@article{Pedrotti:2026PostModes,
    author = "Pedrotti, Davide and Giar\`{e}, William and Cheng, Hanyu and Di Valentino, Eleonora",
    title = "{When, Why and How CMB compression Fails}",
    eprint = "2610.nnnnn",
    archivePrefix = "arXiv",
    year = "2026",
    note = "Provisional citation: the arXiv identifier and citation key will be updated after publication on arXiv"
}
```

</details>

The arXiv identifier and citation key are provisional and will be updated once
the paper is available on arXiv. The BibTeX entry is also provided in
[CITATION.bib](CITATION.bib).
