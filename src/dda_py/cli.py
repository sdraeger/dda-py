"""DDA command-line interface.

Usage::

    dda --file data.edf --channels 0 1 2 --variants ST --wl 200 --ws 100
"""
from __future__ import annotations

import argparse
import json
import sys

from .runner import DDARunner, DDARequest, Defaults
from .variants import DEFAULT_DELAYS


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dda",
        description="Run DDA (Delay Differential Analysis) on a data file.",
    )
    parser.add_argument("--file", required=True, help="Input data file path (EDF or ASCII)")
    parser.add_argument("--channels", nargs="+", type=int, required=True, help="0-based channel indices")
    parser.add_argument("--variants", nargs="+", default=["ST"], help="Variant abbreviations (ST, CT, CD, DE, SY)")
    parser.add_argument("--wl", type=int, default=Defaults.WINDOW_LENGTH, help="Window length in samples")
    parser.add_argument("--ws", type=int, default=Defaults.WINDOW_STEP, help="Window step in samples")
    parser.add_argument("--delays", nargs="+", type=int, default=list(DEFAULT_DELAYS), help="Delay values (tau)")
    parser.add_argument("--model", nargs="+", type=int, default=Defaults.MODEL_PARAMS, help="Model encoding indices")
    parser.add_argument("--binary", default=None, help="Path to DDA binary (auto-discovered if omitted)")
    parser.add_argument("--dm", type=int, default=Defaults.MODEL_DIMENSION, help="Model dimension")
    parser.add_argument("--order", type=int, default=Defaults.POLYNOMIAL_ORDER, help="Polynomial order")
    parser.add_argument("--nr-tau", type=int, default=Defaults.NUM_TAU, help="Number of tau values")
    parser.add_argument("--ct-wl", type=int, default=None, help="CT-specific window length")
    parser.add_argument("--ct-ws", type=int, default=None, help="CT-specific window step")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file (default: stdout)")

    args = parser.parse_args()

    runner = DDARunner(binary_path=args.binary)
    request = DDARequest(
        file_path=args.file,
        channels=args.channels,
        variants=args.variants,
        window_length=args.wl,
        window_step=args.ws,
        delays=args.delays,
        model_params=args.model,
        model_dimension=args.dm,
        polynomial_order=args.order,
        num_tau=args.nr_tau,
        ct_window_length=args.ct_wl,
        ct_window_step=args.ct_ws,
    )

    results = runner.run(request)

    output = json.dumps(results, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
