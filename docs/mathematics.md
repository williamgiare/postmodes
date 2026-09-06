# Mathematics

This page summarizes the mathematical backbone of `posterior-eigenmodes`.

The package compares two posterior distributions defined on the same physical parameter space:

- `A`: reference posterior,
- `B`: alternative posterior.

The goal is not only to ask whether one posterior is broader than the other, but to decompose the difference into:

- mean shifts,
- rotations of principal directions,
- direction-dependent broadening or tightening,
- isotropic versus anisotropic deformation.

## 1. Single-Posterior Geometry

For a posterior with covariance matrix `C`, the eigen-decomposition

```math
C = W \Lambda W^T
```

defines:

- eigenvectors `W`: principal directions of the posterior ellipsoid,
- eigenvalues `\Lambda`: variances along those directions.

So for a single posterior:

- axis orientation comes from the eigenvectors,
- axis lengths come from the square roots of the eigenvalues.

This is the covariance-ellipsoid approximation to the posterior. It describes
second moments exactly when they exist, but does not describe all non-Gaussian
features. Raw PCA axes and eigenvalues depend on parameter units. The square
roots are semi-axis lengths at Mahalanobis radius one, not full axis lengths
or the boundary of a 68% probability region in multiple dimensions.

## 2. Mean Shifts

If the posterior means are `\mu_A` and `\mu_B`, define

```math
\Delta \mu = \mu_B - \mu_A.
```

The package reports three covariance-weighted distances:

```math
S_{B|A} = \sqrt{\Delta\mu^T C_A^{-1} \Delta\mu},
```

```math
S_{A|B} = \sqrt{\Delta\mu^T C_B^{-1} \Delta\mu},
```

```math
S_{AB} = \sqrt{\Delta\mu^T (C_A + C_B)^{-1} \Delta\mu}.
```

Their interpretation is geometric:

- `S_{B|A}`: how far `B` is from `A` in the internal units of `A`,
- `S_{A|B}`: how far `A` is from `B` in the internal units of `B`,
- `S_{AB}`: symmetric combined-covariance distance.

These are geometric distances. Even for independent Gaussian estimates, a
multidimensional Mahalanobis radius is not directly a one-dimensional Gaussian
significance: its squared value requires a calibrated reference distribution.
For estimates from shared data, the covariance of their difference also involves
cross-covariances. The code does not estimate these or assign tension p-values.

## 3. Rotation Between Two Posteriors

Version 1.0.0 uses a common dimensionless metric by default. Define

```math
D_A=\operatorname{diag}(\sqrt{(C_A)_{ii}}),\qquad
\widehat C_A=D_A^{-1}C_AD_A^{-1},\qquad
\widehat C_B=D_A^{-1}C_BD_A^{-1}.
```

Let `w_i^{(A)}` and `w_j^{(B)}` be the eigenvectors of these two standardized
matrices. Both use the same scales from A. These are different from the raw PCA
axes printed in report sections A and B. Standardization preserves correlations;
it does not whiten A into a sphere.

A convenient rotation diagnostic is the overlap matrix

```math
O_{ij} = \left| \left(w_i^{(A)}\right)^T w_j^{(B)} \right|.
```

Interpretation:

- values near `1`: strong alignment,
- values near `0`: near-orthogonality,
- off-diagonal structure: mixing between modes.

This quantifies alignment in reference-sigma units. It is invariant to common
changes of parameter units, but not arbitrary reparametrizations. Use
`rotation_basis="original"` for the previous coordinate-dependent convention.
An overlap is an unsigned axis cosine; eigenvector signs are arbitrary, and
sorting eigenvalues can permute mode labels. The overlap matrix is not itself
a signed rotation matrix. There is no unique scalar rotation angle in n dimensions.

Adjacent eigenvalue ratios at most 1.1 trigger a heuristic degeneracy note when
interpretation is enabled. Individual axes inside a degenerate cluster are not
identifiable. `rotation_subspace_angles` compares complete selected clusters in
the declared metric. Their stability requires separation from the remaining
spectrum. Comparing full n-dimensional bases always yields zero subspace angles.

## 4. Whitened Comparison Matrix

To compare `B` against `A`, the package whitens with respect to `A`:

```math
C = C_A^{-1/2} C_B C_A^{-1/2}.
```

Geometric meaning:

- in whitened coordinates, `A` becomes the identity,
- `B` becomes a deformation relative to that unit sphere.

So `C` is the central comparison matrix. It isolates the geometry of the change from `A` to `B`.

Numerically, the solver scales both inputs by D_A, uses Cholesky whitening, and
uses the orthogonal polar factor to recover the displayed symmetric-whitening
convention. The definition of C has not changed. Both covariances must be positive
definite; singular inputs are rejected without eigenvalue clipping or regularization.
`eigenvalue_floor` is now a relative min/max spectral tolerance in correlation
coordinates (default 1e-14). Symmetry is also checked in dimensionless coordinates.

## 5. Generalized Eigenvalues

Diagonalizing `C`,

```math
C = U D U^T,
```

gives generalized eigenvalues `\rho_i` in `D`.

These are mode-by-mode variance ratios:

```math
\rho_i = \frac{\mathrm{Var}_B(m_i)}{\mathrm{Var}_A(m_i)}.
```

Interpretation:

- `\rho_i > 1`: `B` is broader than `A` along mode `i`,
- `\rho_i < 1`: `B` is tighter than `A`,
- `\rho_i = 1`: the variance is preserved.

The package also reports:

```math
\sqrt{\rho_i},
```

which is the ratio of standard deviations along the same mode.

## 6. Isotropic and Anisotropic Deformation

If the deformation from `A` to `B` were the same in every direction, the comparison matrix would be

```math
C = \alpha I.
```

This motivates an isotropic scale factor:

```math
\alpha = (\det C)^{1/N}
= \exp\!\left[\frac{1}{N}\sum_i \log \rho_i\right].
```

Interpretation:

- `\alpha > 1`: `B` is broader than `A` on average,
- `\alpha < 1`: `B` is tighter than `A` on average,
- `\alpha = 1`: no average isotropic rescaling.

To quantify the residual shape distortion, the package uses

```math
A_{\mathrm{aniso}} =
\sqrt{\frac{1}{N}\sum_i \left(\log \rho_i - \overline{\log \rho}\right)^2}.
```

Interpretation:

- `A_{\mathrm{aniso}} = 0`: perfectly isotropic deformation,
- `A_{\mathrm{aniso}} > 0`: anisotropic deformation.

So the deformation is decomposed into:

- overall scale: `\alpha`,
- residual shape distortion: `A_{\mathrm{aniso}}`.

All logarithms are natural. Alpha is a variance scale, sqrt(alpha) a linear scale.
The covariance-ellipsoid volume ratio is alpha^(N/2). Alpha=1 means equal volumes,
not equal shapes. A rotation of an anisotropic ellipsoid can itself generate
non-unit rho and nonzero A_aniso. These metrics do not uniquely separate physical
causes such as loss of likelihood information and changes induced by priors.

## 7. Generalized Modes

The eigenvectors of the whitened comparison matrix live in whitened coordinates. To interpret them in the original parameter basis, they are mapped back to parameter space.

This yields generalized modes:

```math
m_i = v_i^T x,
```

where:

- `x` is the parameter vector,
- `v_i` is the comparison mode in the original coordinates.

These modes are not the PCA modes of `A` or `B` separately. They are the directions that are most informative for the difference between `A` and `B`.

Precisely, V=C_A^(-1/2)U contains projection coefficients, with

```math
V^T C_A V=I,\qquad V^T C_B V=\operatorname{diag}(\rho_i).
```

Default projections are `(x-mu_A)^T V` for both datasets. A has unit covariance
when projected samples reproduce the input covariance; B retains its mean shift.
`center="separate"` removes each mean, while `center="none"` applies no centering.
`normalization="euclidean"` rescales each column to Euclidean norm one. This changes
absolute variances but preserves B/A variance ratios. Reports display this latter
normalization for readability; coefficients are unit-dependent, not parameter importance.

Projection coefficients are covectors, not displacement axes of the original
ellipsoid. The corresponding displacement at fixed other mode coordinates is a
column of V^(-T)=C_A^(1/2)U. Generalized modes need not be Euclidean-orthogonal in
the original coordinates. Zero cross-covariance does not imply independence for
non-Gaussian posteriors. Use sample weights for all empirical distributions and variances.

## 8. What the Package Measures

Putting everything together, the package measures:

- PCA structure of `A`,
- PCA structure of `B`,
- mean shifts between `A` and `B`,
- rotation of principal directions,
- generalized variance ratios `\rho_i`,
- isotropic scale `\alpha`,
- anisotropy `A_{\mathrm{aniso}}`,
- generalized mode directions,
- correlation-structure changes in the original parameter basis.

This is the mathematical core of `posterior-eigenmodes`.
