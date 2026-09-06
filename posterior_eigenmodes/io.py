from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from getdist.mcsamples import loadMCSamples


def load_covmat(path: str | Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load a Cobaya-style covariance matrix and its parameter names."""

    covmat_path = Path(path)
    with covmat_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()

    if not header.startswith("#"):
        raise ValueError(
            "Covariance matrix file must start with a header line beginning with '#'."
        )

    parameter_names = tuple(header[1:].strip().split())
    if not parameter_names:
        raise ValueError("Covariance matrix header does not contain parameter names.")

    matrix = np.loadtxt(covmat_path, comments="#", dtype=float, ndmin=2)
    if matrix.ndim != 2:
        raise ValueError(
            "Loaded covariance matrix must be a 2D array; "
            f"received shape {matrix.shape}."
        )
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Loaded covariance matrix must be square; "
            f"received shape {matrix.shape}."
        )
    if len(parameter_names) != matrix.shape[0]:
        raise ValueError(
            "Number of parameter names in the header must match matrix dimension; "
            f"received {len(parameter_names)} names for dimension {matrix.shape[0]}."
        )

    return matrix, parameter_names


def load_getdist_chain(
    path_or_root: str | Path,
    *,
    settings: Mapping[str, Any] | None = None,
):
    """Load a Cobaya/GetDist chain from its file root.

    By default, the first 30% of rows are ignored to match the intended
    real-chain workflow.
    """

    chain_settings = {"ignore_rows": 0.3}
    if settings is not None:
        chain_settings.update(dict(settings))

    return loadMCSamples(str(path_or_root), no_cache=True, settings=chain_settings)
