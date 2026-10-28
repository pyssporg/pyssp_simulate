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
import hashlib
import json
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

#: Provenance file written next to the FMUs. It pins both the FMU bytes and
#: the model sources they were built from, so a cache that no longer matches
#: the tree fails loudly instead of being reused.
MANIFEST_NAME = "FMU_MANIFEST.json"
MANIFEST_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest() -> str:
    """Digest of the Modelica model and the omc build script."""
    digest = hashlib.sha256()
    for name in sorted(SCRIPT_NAMES):
        digest.update(name.encode())
        digest.update((FMU_SOURCE_DIR / name).read_bytes())
    return digest.hexdigest()


def write_manifest(output_dir: Path) -> Path:
    """Write the provenance manifest for the FMUs in ``output_dir``."""
    payload = {
        "version": MANIFEST_VERSION,
        "source_digest": source_digest(),
        "fmus": {
            name: {
                "sha256": _sha256(output_dir / name),
                "size": (output_dir / name).stat().st_size,
            }
            for name in FMU_NAMES
        },
    }
    path = output_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def verify_manifest(output_dir: Path) -> list[str]:
    """Return the ways ``output_dir`` fails to match its manifest.

    An empty list means the cached FMUs are exactly the ones recorded for
    the current model sources. Problems are described, not raised, so the
    caller decides whether to fail or rebuild.
    """
    path = output_dir / MANIFEST_NAME
    if not path.is_file():
        return [
            f"missing {MANIFEST_NAME} in {output_dir} "
            f"(regenerate it with tests/generate_fmus.py --output {output_dir})"
        ]
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return [f"unreadable {MANIFEST_NAME} in {output_dir}: {error}"]

    problems: list[str] = []
    if payload.get("version") != MANIFEST_VERSION:
        problems.append(
            f"manifest version {payload.get('version')!r} != {MANIFEST_VERSION}"
        )
    if payload.get("source_digest") != source_digest():
        problems.append("FMUs were built from different model sources")
    recorded = payload.get("fmus", {})
    for name in FMU_NAMES:
        entry = recorded.get(name)
        fmu_path = output_dir / name
        if entry is None:
            problems.append(f"no manifest entry for {name}")
        elif not fmu_path.is_file():
            problems.append(f"{name} is missing")
        elif _sha256(fmu_path) != entry.get("sha256"):
            problems.append(f"{name} does not match its recorded digest")
    return problems


def generate_fmus(output_dir: Path, force: bool = False) -> list[Path]:
    """Compile the test FMUs with omc and return their paths.

    Copies ``TestFMUs.mo`` and ``build_fmus.mos`` into ``output_dir``
    and runs ``omc build_fmus.mos`` with that directory as the working
    directory. When both FMUs and a matching manifest already exist
    there (and ``force`` is false) nothing is compiled and the existing
    files are returned.

    :param output_dir: directory to build in (created when missing).
    :param force: rebuild even when a valid cache already exists.
    :raises RuntimeError: when the cached FMUs do not match their
        manifest, or when omc is not installed or the build fails; the
        message then includes omc's output.
    """
    paths = [output_dir / name for name in FMU_NAMES]
    if not force and all(path.is_file() for path in paths):
        problems = verify_manifest(output_dir)
        if not problems:
            return paths
        if (output_dir / MANIFEST_NAME).is_file():
            raise RuntimeError(
                f"stale FMU cache in {output_dir}: " + "; ".join(problems)
            )

    omc = shutil.which(OMC)
    if omc is None:
        raise RuntimeError(
            f"the OpenModelica compiler ({OMC!r}) was not found on PATH; install "
            "OpenModelica, or point PYSSP_SIMULATE_TEST_FMU_DIR at a directory "
            f"containing {' and '.join(FMU_NAMES)} and a valid {MANIFEST_NAME}"
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
    write_manifest(output_dir)
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
