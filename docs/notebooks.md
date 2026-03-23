# Notebooks

The repository includes a small set of notebooks with different roles.

## 1_Theory.ipynb

Theory-first notebook explaining:

- PCA of single posteriors,
- shifts,
- rotations,
- whitening,
- generalized eigenvalues,
- isotropic vs anisotropic deformation,
- generalized modes.

This is the best place to understand the geometry behind the package.

## 2_Validation.ipynb

Synthetic validation notebook with controlled covariance matrices.

Use it to check whether:

- rotations are recovered correctly,
- generalized eigenvalues are recovered correctly,
- projection effects are understood correctly.

## 3_Basic_Example.ipynb

Real-data workflow example using:

- Cobaya covariance matrices,
- Cobaya/GetDist chains,
- `eigenmodes(...)`,
- `eigenmode_report(...)`,
- mode-space plots.

## 4_Advanced_Example.ipynb

Extended real-data example showing:

- the full structured report,
- focused diagnostics for shifts, deformation, and correlation changes,
- eigenvector-overlap plots,
- generalized mode summaries,
- mode-space histograms and GetDist plots.
