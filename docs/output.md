# Output

This page explains how to read the main quantities reported by `eigenmode_report(...)`.

## Section `A`

This is the PCA decomposition of the reference posterior.

For each mode:

- `lambda`: variance along that PCA direction,
- `sigma`: standard deviation along that direction,
- `direction`: linear combination of the original parameters.

Interpretation:

- large `lambda`: broad direction,
- small `lambda`: tightly constrained direction.

## Section `B`

This is the PCA decomposition of the alternative posterior, with the same interpretation as for `A`.

## Section `Shifts`

This block reports:

- `delta mu = mu_B - mu_A`
- `B shifted from A in A units`
- `A shifted from B in B units`
- `joint shift in combined units`

Interpretation:

- large `B shifted from A in A units`: `B` is far from `A` relative to the scale of `A`,
- large `A shifted from B in B units`: `A` is far from `B` relative to the scale of `B`,
- `joint shift in combined units`: symmetric geometric distance.

These are geometric distances, not automatically formal tension significances when `A` and `B` are statistically correlated.

## Section `Rotations`

This block reports overlaps between PCA modes of `A` and `B`.

Interpretation:

- overlap near `1`: almost the same principal direction,
- overlap near `0`: orthogonal directions,
- large off-diagonal overlaps: mixing between modes.

The final interpretation line summarizes whether:

- the principal bases are almost unchanged,
- mildly mixed,
- or substantially rotated.

## Section `C = C_A^(-1/2) C_B C_A^(-1/2)`

This is the main comparison section.

### Comparison Matrix

The printed matrix `C` is the whitened comparison matrix itself.

If `C = I`, then `B` has the same covariance geometry as `A` after whitening.

### Comparison Diagnostics

This section reports the condition numbers of `C_A` and `C_B`.

Interpretation:

- small condition number: numerically stable covariance,
- large condition number: highly anisotropic covariance,
- very large condition number: smallest modes should be interpreted with extra caution.

### Isotropic vs Anisotropic Deformation

- `alpha`: average isotropic rescaling
- `A_aniso`: anisotropic shape distortion

Interpretation:

- `alpha > 1`: `B` broader than `A` on average,
- `alpha < 1`: `B` tighter than `A` on average,
- `A_aniso = 0`: perfectly isotropic deformation,
- large `A_aniso`: strongly direction-dependent deformation.

### Degradation Summary

- `rho`: variance ratios along generalized modes
- `sqrt(rho)`: standard-deviation ratios

Interpretation:

- `rho > 1`: `B` broader than `A` along that mode,
- `rho < 1`: `B` tighter than `A`,
- `rho ~= 1`: approximately unchanged.

The report also summarizes:

- how many modes are degraded,
- how many are improved,
- which mode changes the most.

### Generalized Modes

Each generalized mode is a parameter combination selected to describe the change from `A` to `B`.

For each mode the report gives:

- `rho`
- `sigma ratio`
- interpretation
- direction in the original parameters

These are the main scientific output for comparison.

### Largest Correlation Changes

This is the comparison viewed back in the original parameter basis.

Interpretation:

- large changes identify parameter pairs whose degeneracy structure changes the most,
- this can be visually useful when relating generalized-mode results back to familiar 2D posterior plots.
