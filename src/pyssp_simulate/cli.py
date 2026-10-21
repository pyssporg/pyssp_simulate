"""Command line interface: ``pyssp-simulate``.

Subcommands:

* ``run`` -- simulate an SSP package with one backend.
* ``backends`` -- list the registered backends and their availability.
* ``compare`` -- compare result CSVs produced by ``run`` (or any engine).
* ``run-all`` -- run every permutation described by a JSON config file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyssp_simulate.compare import ComparisonWindow, compare, compare_to_reference
from pyssp_simulate.interface import (
    BackendNotAvailableError,
    available_backends,
    backend_names,
    get_simulator_class,
    simulate,
)
from pyssp_simulate.permutations import load_permutation_config, run_permutations
from pyssp_simulate.result import SimulationResult
from pyssp_simulate.spec import SimulationSpec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyssp-simulate",
        description="Simulate SSP packages with different simulation engines "
        "and compare their results.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Simulate an SSP package")
    run_parser.add_argument("ssp", help=".ssp archive or unpacked SSP directory")
    run_parser.add_argument(
        "--backend",
        default="auto",
        help=f"simulation backend: {', '.join(backend_names())}, or auto (default)",
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

    compare_parser = subparsers.add_parser(
        "compare", help="Compare result CSVs (first file is the reference)"
    )
    compare_parser.add_argument(
        "csvs", nargs="*", help="result CSV files (two or more without --reference)"
    )
    compare_parser.add_argument(
        "--reference",
        default=None,
        help="trusted reference CSV; each positional CSV is compared against it",
    )
    compare_parser.add_argument(
        "--model-name",
        default=None,
        help="strip 'default.<model-name>.' from reference CSV headers",
    )
    compare_parser.add_argument(
        "--strip-prefix",
        default=None,
        help="strip this exact prefix from every result CSV header (for example 'fmu.')",
    )
    compare_parser.add_argument(
        "--signals", default=None, help="comma-separated subset of signals to compare"
    )
    compare_parser.add_argument(
        "--atol", type=float, default=1e-6, help="absolute tolerance for pass/fail"
    )
    compare_parser.add_argument(
        "--rtol", type=float, default=0.0, help="relative tolerance for pass/fail"
    )
    compare_parser.add_argument(
        "--start-time", type=float, default=None, help="explicit comparison start time"
    )
    compare_parser.add_argument(
        "--stop-time", type=float, default=None, help="explicit comparison stop time"
    )
    compare_parser.add_argument(
        "--step-size", type=float, default=None, help="explicit comparison step size"
    )
    compare_parser.add_argument(
        "--output", "-o", default=None, help="write per-signal metrics to this CSV"
    )
    compare_parser.add_argument(
        "--json", action="store_true", help="print the summary as JSON"
    )

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


def _comparison_window(args: argparse.Namespace) -> ComparisonWindow | None:
    values = (args.start_time, args.stop_time, args.step_size)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "explicit comparison grid needs --start-time, --stop-time and --step-size"
        )
    return ComparisonWindow(
        start_time=args.start_time,
        stop_time=args.stop_time,
        step_size=args.step_size,
    )


def _compare_command(args: argparse.Namespace) -> int:
    if args.reference is None and len(args.csvs) < 2:
        print(
            "error: compare needs at least two CSV files without --reference",
            file=sys.stderr,
        )
        return 2
    if args.reference is not None and not args.csvs:
        print("error: --reference needs at least one result CSV", file=sys.stderr)
        return 2

    signals = args.signals.split(",") if args.signals else None
    window = _comparison_window(args)
    results = [
        SimulationResult.from_csv(
            path,
            backend=Path(path).stem,
            strip_prefix=args.strip_prefix,
        )
        for path in args.csvs
    ]

    if args.reference is not None:
        report = compare_to_reference(
            args.reference,
            results,
            model_name=args.model_name,
            reference_backend=Path(args.reference).stem,
            signals=signals,
            window=window,
        )
    else:
        report = compare(results, signals=signals, window=window)

    if args.json:
        print(json.dumps(report.summary(), indent=2))
    else:
        for pair in report.pairs:
            print(
                f"{pair.reference_backend} vs {pair.candidate_backend}: "
                f"max_abs_error={pair.max_abs_error:.6g} "
                f"rmse={pair.rmse:.6g} signals={len(pair.comparisons)}"
            )

    if args.output:
        report.write_csv(args.output)
        print(f"metrics written to {args.output}")

    if report.passed(atol=args.atol, rtol=args.rtol):
        if not args.json:
            print(f"PASS (atol={args.atol}, rtol={args.rtol})")
        return 0
    print(f"FAIL (atol={args.atol}, rtol={args.rtol})")
    return 1


def _run_all_command(args: argparse.Namespace) -> int:
    config = load_permutation_config(args.config)
    runs = run_permutations(config)

    if not runs:
        print("error: the config produced zero permutations", file=sys.stderr)
        return 1

    compared = passed = failed = 0
    for index, run in enumerate(runs):
        backends = ",".join(run.results) or "none"
        skipped = ",".join(run.skipped) or "none"
        if run.report is not None:
            compared += 1
            max_error = max(
                (pair.max_abs_error for pair in run.report.pairs), default=0.0
            )
            error_text = f"max_abs_error={max_error:.6g}"
        else:
            error_text = "max_abs_error=n/a"
        if run.passed is True:
            passed += 1
            status = "PASS"
        elif run.passed is False or not run.results:
            failed += 1
            status = "FAIL"
        else:
            status = "ok"
        print(
            f"perm_{index:04d}: backends={backends} skipped={skipped} "
            f"{error_text} [{status}]"
        )

    print(
        f"total permutations={len(runs)}, compared={compared}, "
        f"passed={passed}, failed={failed}"
    )
    if failed:
        print(f"error: {failed} of {len(runs)} permutation(s) failed", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "run": _run_command,
        "backends": _backends_command,
        "compare": _compare_command,
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
