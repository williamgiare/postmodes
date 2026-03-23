# Validation

The package includes a synthetic validation notebook '2_Validation.ipynb'

The purpose of that notebook is to verify that the code recovers known geometric inputs in controlled toy models.

## Validation Cases

The notebook currently covers:

1. **Pure PCA rotation**
   - same eigenvalues,
   - rotated PCA basis,
   - validates overlap-based rotation diagnostics.

2. **Pure generalized stretch/compression**
   - direct control over the comparison matrix,
   - validates recovery of the generalized eigenvalues `rho_i`.

3. **Mixed controlled case**
   - non-trivial PCA geometry,
   - controlled comparison matrix,
   - validates PCA structure and generalized comparison simultaneously.

4. **Projection effect in 3D**
   - 2D marginals can change strongly,
   - while the full 3D eigenbasis remains unchanged,
   - validates the distinction between projected contour changes and genuine `n`-dimensional basis rotation.

## Why This Matters

The validation notebook is useful because it helps distinguish:

- genuine scientific behavior in real chains,
- from numerical or implementation mistakes,
- from misleading visual intuition based only on 2D projected contours.
