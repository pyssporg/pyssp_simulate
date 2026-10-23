"""Command line interface: ``pyssp-simulate``.

Subcommands:

* ``run`` -- simulate an SSP package with one backend.
* ``backends`` -- list the registered backends and their availability.
* ``run-all`` -- run every permutation described by a JSON config file.

Comparison is not a subcommand here. Measurement and judgment live in
``pyssp_vv``; ``run-all`` reports metrics when two or more backends produce a
result, but it does not decide whether they agree.
"""

from __future__ import annotations

import argparse
import sys

from pyssp_simulate.interface import (
    DEFAULT_BACKEND,
    BackendNotAvailableError,
    available_backends,
    backend_names,
    get_simulator_class,
    simulate,
)
from pyssp_simulate.permutations import load_permutation_config, run_permutations
from pyssp_simulate.spec import SimulationSpec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyssp-simulate",
        description="Simulate SSP packages with different simulation engines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Simulate an SSP package")
    run_parser.add_argument("ssp", help=".ssp archive or unpacked SSP directory")
    run_parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        type=str.lower,
        choices=backend_names(),
        help=f"simulation backend (default: {DEFAULT_BACKEND})",
    )
    run_parser.add_argument("--start-time", type=float, default=0.0, help="start time")
    run_parser.add_argument(
        "--stop-time", type=float, default=1.0, help="stop time (default 1.0)"
    )
    run_parser.add_argument(
        "--step-size", type=float, default=None, help="communication/logging step size"
    )
    run_parser.add_argument(
        "--tolerance", type=float, default=None, help="solver/iteration tolerance"
    )
    run_parser.add_argument(
        "--output", "-o", default=None, help="write the result CSV to this path"
    )

    subparsers.add_parser("backends", help="List backends and their availability")

    run_all_parser = subparsers.add_parser(
        "run-all", help="Run every permutation described by a JSON config file"
    )
    run_all_parser.add_argument(
        "config", help="path to the permutation config JSON file"
    )

    return parser


def _run_command(args: argparse.Namespace) -> int:
    spec = SimulationSpec(
        ssp=args.ssp,
        start_time=args.start_time,
        stop_time=args.stop_time,
        step_size=args.step_size,
        tolerance=args.tolerance,
        result_file=args.output,
    )
    result = simulate(spec, backend=args.backend)
    runtime = f"{result.runtime_s:.3f}s" if result.runtime_s is not None else "n/a"
    print(
        f"backend={result.backend} signals={len(result.signals)} "
        f"samples={len(result)} runtime={runtime}"
    )
    if args.output:
        print(f"result written to {args.output}")
    return 0


def _backends_command(_args: argparse.Namespace) -> int:
    for name, (ok, reason) in available_backends().items():
        hint = get_simulator_class(name).install_hint
        if ok:
            print(f"{name:12} available")
        else:
            print(f"{name:12} unavailable ({reason})")
            if hint:
                print(f"{'':12} {hint}")
    return 0


def _run_all_command(args: argparse.Namespace) -> int:
    config = load_permutation_config(args.config)
    runs = run_permutations(config)

    if not runs:
        print("error: the config produced zero permutations", file=sys.stderr)
        return 1

    measured = 0
    missing = 0
    for index, run in enumerate(runs):
        backends = ",".join(run.results) or "none"
        skipped = ",".join(run.skipped) or "none"
        if run.measurements is not None:
            measured += 1
            worst = max(
                (
                    signal.max_abs_error
                    for item in run.measurements
                    for signal in item.signals
                ),
                default=0.0,
            )
            error_text = f"max_abs_error={worst:.6g}"
        else:
            missing += 1
            error_text = "max_abs_error=n/a"
        print(
            f"perm_{index:04d}: backends={backends} skipped={skipped} {error_text}"
        )

    print(
        f"total permutations={len(runs)}, measured={measured}, "
        f"without_measurement={missing}"
    )
    if missing:
        print(
            f"note: {missing} permutation(s) produced no measurement "
            "(fewer than two backends, or pyssp-vv is not installed)",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "run": _run_command,
        "backends": _backends_command,
        "run-all": _run_all_command,
    }
    try:
        return handlers[args.command](args)
    except (BackendNotAvailableError, FileNotFoundError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
