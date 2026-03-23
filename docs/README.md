# Overview

`posterior-eigenmodes` is a small Python package for comparing posterior constraints through the geometry of their covariance matrices.

The package is designed for the following scientific question:

> given two posterior distributions on the same physical parameter space, how do their widths, degeneracy directions, and physically relevant parameter combinations differ?

The main use case is:

- `A`: a reference posterior,
- `B`: an alternative posterior,
- same physical parameters, but possibly different parameter names in the files,
- inputs given either as Cobaya covariance matrices or as Cobaya/GetDist chains.

The package is covariance-driven:

- if covariance matrices are already available, they are used directly,
- if only chains are available, means and covariances are reconstructed from the weighted samples,
- if both are available, the covariance analysis uses the supplied covariance matrices while posterior means can still be estimated from the chains for shift diagnostics.

The central comparison object is

```math
C = C_A^{-1/2} C_B C_A^{-1/2},
```

where:

- `C_A` is the covariance of the reference posterior,
- `C_B` is the covariance of the alternative posterior.

This matrix answers a clean geometric question:

> once the internal scale and orientation of `A` have been factored out, what does `B` look like relative to `A`?

From this construction the package extracts:

- PCA modes of `A`,
- PCA modes of `B`,
- generalized comparison modes of `C`,
- mean-shift diagnostics,
- rotation diagnostics,
- mode-by-mode variance ratios,
- isotropic and anisotropic deformation summaries,
- correlation-structure changes.

The package therefore separates the comparison into a few distinct geometric questions:

1. are the posterior centers shifted?
2. do the principal directions rotate?
3. along which directions does `B` broaden or tighten relative to `A`?
4. is the overall deformation mostly isotropic, or strongly anisotropic?

## Documentation Map

- `mathematics.md`: mathematical definitions
- `api.md`: public API
- `input.md`: input patterns and parameter matching
- `output.md`: how to read the output quantities
- `validation.md`: synthetic validation strategy
- `notebooks.md`: notebook guide