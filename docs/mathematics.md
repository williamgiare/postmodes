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

This is the PCA geometry of the posterior.

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

These are distance measures in posterior units. They should not automatically be interpreted as formal tension significances if `A` and `B` are statistically correlated.

## 3. Rotation Between Two Posteriors

Let `w_i^{(A)}` be the eigenvectors of `C_A` and `w_j^{(B)}` the eigenvectors of `C_B`.

A convenient rotation diagnostic is the overlap matrix

```math
O_{ij} = \left| \left(w_i^{(A)}\right)^T w_j^{(B)} \right|.
```

Interpretation:

- values near `1`: strong alignment,
- values near `0`: near-orthogonality,
- off-diagonal structure: mixing between modes.

This quantifies how much the principal bases of `A` and `B` rotate relative to each other.

## 4. Whitened Comparison Matrix

To compare `B` against `A`, the package whitens with respect to `A`:

```math
C = C_A^{-1/2} C_B C_A^{-1/2}.
```

Geometric meaning:

- in whitened coordinates, `A` becomes the identity,
- `B` becomes a deformation relative to that unit sphere.

So `C` is the central comparison matrix. It isolates the geometry of the change from `A` to `B`.

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
