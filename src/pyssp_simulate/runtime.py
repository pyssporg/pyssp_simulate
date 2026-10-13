"""Runtime preparation of SSP packages for the simulation backends.

Backends have different needs: some want an ``.ssp`` archive
(OMSimulator, FMPy, cosim), one wants an unpacked directory tree
(ssp4sim). Unpacked trees may also contain *unpacked* FMUs (a directory
with a ``modelDescription.xml``), which engines cannot load, so they are
repackaged into ``.fmu`` archives. This module normalizes any of these
inputs into a :class:`RuntimeSSP` that provides both shapes.
"""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSSP:
    """A normalized SSP package ready for simulation.

    :param archive: path to a runnable ``.ssp`` zip archive.
    :param root: path to the unpacked package tree.
    :param ssd_name: file name of the active SSD variant.
    :param root_system: name of the SSD's root system (used to canonicalize
        signal names), or ``None`` when it cannot be determined.
    """

    archive: Path
    root: Path
    ssd_name: str
    root_system: str | None


def read_root_system_name(ssd_path: Path) -> str | None:
    """Return the name of the SSD root system, or ``None`` if unreadable.

    Uses a plain XML parse (no schema validation) so exotic but valid
    SSD files still work.
    """
    try:
        tree = ET.parse(ssd_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "System":
            return element.get("name")
    return None


def package_fmu_dir(fmu_dir: Path, fmu_path: Path) -> None:
    """Zip an unpacked FMU directory into an ``.fmu`` archive."""
    with zipfile.ZipFile(fmu_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(fmu_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(fmu_dir))


def repackage_unpacked_fmus(root: Path) -> list[Path]:
    """Turn unpacked FMU directories inside ``root`` into ``.fmu`` archives.

    A directory is treated as an unpacked FMU when it directly contains a
    ``modelDescription.xml``. Nested candidates are handled innermost-first:
    each unpacked FMU directory becomes an ``.fmu`` archive in place, so an
    FMU nested inside another ends up as an archive inside its parent's
    archive. Returns the created ``.fmu`` paths.
    """
    candidates = sorted(
        (path.parent for path in root.rglob("modelDescription.xml") if path.is_file()),
        key=lambda path: len(path.parts),
        reverse=True,
    )

    packaged: list[Path] = []
    consumed: list[Path] = []
    for fmu_dir in candidates:
        if any(fmu_dir == done or done in fmu_dir.parents for done in consumed):
            continue
        fmu_path = fmu_dir.parent / (fmu_dir.name + ".fmu")
        package_fmu_dir(fmu_dir, fmu_path)
        shutil.rmtree(fmu_dir)
        consumed.append(fmu_dir)
        packaged.append(fmu_path)
    return packaged


def find_ssd(root: Path) -> Path | None:
    """Locate the active SSD: ``SystemStructure.ssd`` or the first ``.ssd``."""
    default = root / "SystemStructure.ssd"
    if default.is_file():
        return default
    ssd_files = sorted(root.glob("*.ssd"))
    return ssd_files[0] if ssd_files else None


def _zip_tree(root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))


@contextmanager
def prepare_runtime_ssp(source: Path | str) -> Iterator[RuntimeSSP]:
    """Normalize an SSP input into a :class:`RuntimeSSP`.

    Accepts either an ``.ssp`` zip archive or an unpacked SSP directory
    (whose unpacked FMUs are repackaged). The runtime copy lives in a
    temporary directory that is removed when the context exits; nothing
    in the original input is modified.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"SSP path does not exist: {source}")

    with tempfile.TemporaryDirectory(prefix="pyssp_simulate_ssp_") as tmp:
        work = Path(tmp)

        if source.is_file():
            # A ready-made archive: keep it as-is and unpack a copy for
            # backends that want a directory tree.
            archive = work / "runtime.ssp"
            shutil.copy2(source, archive)
            root = work / "ssp"
            with zipfile.ZipFile(source) as zip_file:
                zip_file.extractall(root)
        else:
            root = work / "ssp"
            shutil.copytree(source, root)
            repackage_unpacked_fmus(root)
            archive = work / "runtime.ssp"
            _zip_tree(root, archive)

        ssd_path = find_ssd(root)
        if ssd_path is None:
            raise ValueError(f"No .ssd file found in SSP package: {source}")

        yield RuntimeSSP(
            archive=archive,
            root=root,
            ssd_name=ssd_path.name,
            root_system=read_root_system_name(ssd_path),
        )
