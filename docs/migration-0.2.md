# Version 0.2: numerical and interpretation conventions

The definitions of rho, SCCD, the two reference distances, alpha and A_aniso are
unchanged. Well-resolved positive-definite inputs should recover previous scalar
results to numerical accuracy. Mode signs are arbitrary, and bases inside exactly
degenerate eigenspaces are not unique.

## Intentional changes

- Rotation overlaps now default to a common scaling by A's marginal sigmas.
  The overlap numbers and axes can change; this is a change of metric, not a
  correction to previous generalized eigenvalues. Set `rotation_basis="original"`
  to reproduce old raw-coordinate rotations. A/B report sections still show raw PCA.
- Mode plots default to `normalization="reference", center="reference"`.
  A has unit covariance when the projected sample covariance matches the input;
  B retains its shift. Old behavior is available with `normalization="euclidean",
  center="separate"`. Both conventions preserve weighted variance ratios.
- Reports default to numbers. Add `interpretation=True` to include explanatory
  notes. The degeneracy heuristic is not a statistical uncertainty estimate.
- Singular/indefinite covariances are rejected. There is no silent eigenvalue
  clipping for the log-deformation metrics. The floor and symmetry tolerance are
  dimensionless checks, rather than absolute thresholds in parameter units.
- Full mean vectors are reordered with selected covariances. Use
  `mean_order="selected"` for vectors already permuted into selected order.
- Histograms use sample weights; this can change previously unweighted curves.

## Numerical verification and scientific scope

The solver uses diagonal scaling, Cholesky solves and an orthogonal polar factor
to retain the displayed C=A^(-1/2) B A^(-1/2). Tests compare against an independent
generalized solver, check both projected covariance identities, and stress common
unit changes, parameter permutations, A/B reversal and invalid inputs.

Original-coordinate PCA is computed by an SVD of a scaled Cholesky square root,
avoiding spurious negative small eigenvalues from direct diagonalization of an
extremely poorly scaled covariance. Very large raw condition numbers still make
raw-coordinate axes numerically delicate; they are not a measure of chain quality.

This validates algebra, not the physical adequacy of a likelihood compression.
For a paper, inspect selected parameter definitions, chain convergence and tails,
prior support, burn-in sensitivity, and covariance-file provenance. SCCD describes
center separation geometrically. Alpha and A_aniso describe covariance deformation;
they do not prove equality of full non-Gaussian distributions or a gain in valid
physical information. Comparisons across models also depend on the selected dimension.

Local notebooks not tracked by Git are not automatically migrated.
