"""Generate the FMU test fixtures with the OpenModelica compiler.

The test suite keeps no prebuilt FMU binaries in the repository.
Instead, ``tests/fmus/TestFMUs.mo`` defines two primitive models
(``RampSource`` and ``Integrator``) that are compiled into FMI 2.0
co-simulation FMUs on demand by ``omc``, driven by the script
``tests/fmus/build_fmus.mos``.

Run this module as a script to pre-build the FMUs:

    python tests/generate_fmus.py [--output DIR] [--force]

The default output directory is ``tests/fmus/build/``, which is
git-ignored. On a machine without OpenModelica, build the FMUs
elsewhere and point the ``PYSSP_SIMULATE_TEST_FMU_DIR`` environment
variable at the directory holding them (see ``tests/conftest.py``).

The :func:`generate_fmus` function is also imported by the test suite,
which skips regeneration when both FMUs already exist.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

#: Directory holding the Modelica package and the omc build script.
FMU_SOURCE_DIR = Path(__file__).parent / "fmus"

#: Default output directory (inside the git-ignored ``build/`` tree).
DEFAULT_OUTPUT_DIR = FMU_SOURCE_DIR / "build"

#: Name of the OpenModelica compiler executable.
OMC = "omc"

#: The FMUs produced from TestFMUs.mo, in a stable order.
FMU_NAMES = ("RampSource.fmu", "Integrator.fmu")

#: Files copied into the output directory before running omc.
SCRIPT_NAMES = ("TestFMUs.mo", "build_fmus.mos")


def generate_fmus(output_dir: Path, force: bool = False) -> list[Path]:
    """Compile the test FMUs with omc and return their paths.

    Copies ``TestFMUs.mo`` and ``build_fmus.mos`` into ``output_dir``
    and runs ``omc build_fmus.mos`` with that directory as the working
    directory. When both FMUs already exist there (and ``force`` is
    false) nothing is compiled and the existing files are returned.

    :param output_dir: directory to build in (created when missing).
    :param force: rebuild even when both FMUs already exist.
    :raises RuntimeError: when omc is not installed or the build fails;
        the message then includes omc's output.
    """
    paths = [output_dir / name for name in FMU_NAMES]
    if not force and all(path.is_file() for path in paths):
        return paths

    omc = shutil.which(OMC)
    if omc is None:
        raise RuntimeError(
            f"the OpenModelica compiler ({OMC!r}) was not found on PATH; install "
            "OpenModelica, or point PYSSP_SIMULATE_TEST_FMU_DIR at a directory "
            f"containing {' and '.join(FMU_NAMES)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SCRIPT_NAMES:
        shutil.copy2(FMU_SOURCE_DIR / name, output_dir / name)

    process = subprocess.run(
        [omc, "build_fmus.mos"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (process.stdout + process.stderr).strip()

    missing = [path.name for path in paths if not path.is_file()]
    if process.returncode != 0 or missing:
        raise RuntimeError(
            "omc failed to build the test FMUs"
            + (f" (missing: {', '.join(missing)})" if missing else "")
            + f".\nomc output:\n{output}"
        )
    return paths


def main(argv: list[str] | None = None) -> int:
    """Command line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Build the pyssp_simulate test FMUs with OpenModelica."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when both FMUs already exist in the output directory",
    )
    args = parser.parse_args(argv)

    try:
        paths = generate_fmus(args.output, force=args.force)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"test FMUs in {args.output}:")
    for path in paths:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
