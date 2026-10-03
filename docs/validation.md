# Validation

The package includes a synthetic validation notebook '2_Validation.ipynb'

The purpose of that notebook is to verify that the code recovers known geometric inputs in controlled toy models.

Cases 1, 3 and 4 explicitly use `rotation_basis="original"` because their angles
were designed in the original coordinate plane. New cases test the standardized
metric and invariance under unit changes, as well as weighted projections and
retained shifts. The original ellipse illustrations are preserved.

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

## Numerical regressions

Numerical regression tests also run the solver, weighted covariance estimates,
and mode projections with floating-point exceptions enabled, including 10- and
20-dimensional inputs. Direct NumPy contractions avoid spurious `matmul` warnings
from macOS Accelerate. Non-finite products are rejected; warnings are not globally
suppressed. Results are checked against an independent generalized eigensolver
and both projected covariance identities.

## Sampling stability

The optional `bootstrap_stability` utility resamples circular moving blocks
within each separate chronological chain, retaining row weights. It requires
post-burn-in samples and an explicit block length in stored rows. For compressed
MCMC rows with multiplicities this is a row-process bootstrap: assess stability
across several block lengths, including lengths exceeding relevant correlation
scales. Do not shuffle samples first or join independent chains into one sequence.

For GetDist, split merged samples with `getSeparateChains()`. Pass the result as
a list and supply `parameter_names_a=sample_a.getParamNames().list()` and the
corresponding B names, because GetDist's separate `WeightedSamples` do not carry
parameter-name metadata. Fixed-parameter/derived-column selections must be handled explicitly.
A and B are resampled independently, appropriate for independent MCMC runs even
when they analyze shared observations. The intervals describe Monte Carlo
variability, not observational tension, posterior credible intervals for the
metrics, or uncertainty in supplied covariance files. They assume stationary,
representative sampling; bootstrap cannot repair an unconverged chain.

Spectra are ranked in every replicate, so rho intervals are for ordered eigenvalues,
not matched physical modes at crossings. Failed singular replicates are counted
and warned about rather than silently replaced. Large failure fractions invalidate
an unqualified interpretation of the percentile intervals.
