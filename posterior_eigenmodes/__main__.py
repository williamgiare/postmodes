"""Command-line posterior comparison: python -m posterior_eigenmodes."""

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np

from .api import eigenmodes
from .interpretation import eigenmode_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare posterior means and covariance geometry.")
    parser.add_argument("chain_a", nargs="?", help="Reference GetDist chain root")
    parser.add_argument("chain_b", nargs="?", help="Alternative GetDist chain root")
    parser.add_argument("--covmat-a")
    parser.add_argument("--covmat-b")
    parser.add_argument("--params", nargs="+", help="Same ordered parameters for A and B")
    parser.add_argument("--params-a", nargs="+")
    parser.add_argument("--params-b", nargs="+")
    parser.add_argument("--ignore-rows", type=float, default=0.3, help="Burn-in fraction (default: 0.3)")
    parser.add_argument("--rotation-basis", choices=["reference_standardized", "original"], default="reference_standardized")
    parser.add_argument("--check-covariances", action="store_true")
    parser.add_argument("--interpretation", action="store_true", help="Include concise interpretive notes")
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Emit structured results to stdout")
    args = parser.parse_args(argv)
    if not 0 <= args.ignore_rows < 1:
        parser.error("--ignore-rows must be a fraction in [0, 1).")
    try:
        # GetDist progress goes to stderr, keeping stdout usable for JSON/report files.
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            result = eigenmodes(
                args.chain_a, args.chain_b, covmat_a=args.covmat_a, covmat_b=args.covmat_b,
                params=args.params, params_A=args.params_a, params_B=args.params_b,
                chain_settings={"ignore_rows": args.ignore_rows}, rotation_basis=args.rotation_basis,
                check_covariances=args.check_covariances,
            )
        if args.json:
            data = asdict(result)
            data.update(alpha=result.alpha, A_aniso=result.A_aniso, shifts=result.shifts)
            print(json.dumps(data, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x), allow_nan=False, indent=2))
        else:
            print(eigenmode_report(result, precision=args.precision, interpretation=args.interpretation))
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
