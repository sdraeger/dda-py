"""DDA command-line interface.

Usage::

    dda --file data.edf --channels 1 2 3 --flavors ST --wl 200 --ws 100
"""

from __future__ import annotations

import argparse
import json
import sys

from .runner import DDARequest, DDARunner, Defaults
from .variants import DEFAULT_DELAYS


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dda",
        description="Run DDA (Delay Differential Analysis) on a data file.",
    )
    parser.add_argument(
        "--file", required=True, help="Input data file path (EDF or ASCII)"
    )
    parser.add_argument(
        "--channels", nargs="+", type=int, required=True, help="1-based channel indices"
    )
    parser.add_argument(
        "--flavors", nargs="+", default=None, help="DDA flavors (ST, CT, CD, DE, SY)"
    )
    parser.add_argument(
        "--variants", nargs="+", default=None, help="Compatibility alias for --flavors"
    )
    parser.add_argument(
        "--select",
        nargs="+",
        type=int,
        default=None,
        help="Explicit 6-bit -SELECT mask",
    )
    parser.add_argument(
        "--format",
        choices=["ascii", "edf"],
        default=None,
        help="Override input file type",
    )
    parser.add_argument(
        "--wl",
        type=int,
        default=Defaults.WINDOW_LENGTH,
        help="Window length in samples",
    )
    parser.add_argument(
        "--ws", type=int, default=Defaults.WINDOW_STEP, help="Window step in samples"
    )
    parser.add_argument(
        "--delays",
        nargs="+",
        type=int,
        default=list(DEFAULT_DELAYS),
        help="Delay values (tau)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        type=int,
        default=list(Defaults.MODEL_PARAMS),
        help="Model encoding indices",
    )
    parser.add_argument(
        "--binary", default=None, help="Path to DDA binary (auto-discovered if omitted)"
    )
    parser.add_argument(
        "--dm",
        type=int,
        default=Defaults.MODEL_DIMENSION,
        help="Derivative points passed as -dm",
    )
    parser.add_argument(
        "--order", type=int, default=Defaults.POLYNOMIAL_ORDER, help="Polynomial order"
    )
    parser.add_argument(
        "--nr-tau", type=int, default=Defaults.NUM_TAU, help="Number of tau values"
    )
    parser.add_argument(
        "--ct-wl", type=int, default=None, help="CT-specific window length"
    )
    parser.add_argument(
        "--ct-ws", type=int, default=None, help="CT-specific window step"
    )
    parser.add_argument(
        "--time-range", nargs=2, type=float, default=None, metavar=("START", "STOP")
    )
    parser.add_argument(
        "--sr",
        nargs="+",
        type=int,
        default=None,
        help="Sampling rate: one value or two values",
    )
    parser.add_argument("--out-fn", default=None, help="Binary -OUT_FN base path")
    parser.add_argument("--tau-file", default=None, help="Pass -TAU_file")
    parser.add_argument(
        "--tau2", nargs="+", type=int, default=None, help="Pass -TAU2 values"
    )
    parser.add_argument(
        "--model2", nargs="+", type=int, default=None, help="Pass -MODEL2 values"
    )
    parser.add_argument("--no-norm", action="store_true", help="Pass -NoNorm")
    parser.add_argument(
        "--wn-list", nargs="+", type=int, default=None, help="Pass -WN_list values"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output JSON file (default: stdout)"
    )

    args = parser.parse_args()

    flavors = args.flavors if args.flavors is not None else args.variants
    if flavors is None:
        flavors = ["ST"]
    sampling_rate = None
    if args.sr is not None:
        if len(args.sr) == 1:
            sampling_rate = args.sr[0]
        elif len(args.sr) == 2:
            sampling_rate = tuple(args.sr)
        else:
            parser.error("--sr accepts one value or two values")

    runner = DDARunner(binary_path=args.binary)
    request = DDARequest(
        file_path=args.file,
        channels=args.channels,
        flavors=flavors,
        select=args.select,
        input_format=args.format,
        WL=args.wl,
        WS=args.ws,
        delays=args.delays,
        model=args.model,
        derivative_points=args.dm,
        order=args.order,
        nr_tau=args.nr_tau,
        ct_window_length=args.ct_wl,
        ct_window_step=args.ct_ws,
        time_range=tuple(args.time_range) if args.time_range is not None else None,
        sampling_rate=sampling_rate,
        out_fn=args.out_fn,
        tau_file=args.tau_file,
        tau2=args.tau2,
        model2=args.model2,
        no_norm=args.no_norm,
        WN_list=args.wn_list,
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
