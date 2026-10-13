"""Tests for runtime SSP preparation (archives, unpacked trees, FMU repackaging)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from pyssp_simulate.runtime import (
    find_ssd,
    package_fmu_dir,
    prepare_runtime_ssp,
    read_root_system_name,
    repackage_unpacked_fmus,
)


def make_fmu_dir(parent: Path, name: str = "TestFMU") -> Path:
    fmu_dir = parent / name
    (fmu_dir / "binaries" / "linux64").mkdir(parents=True)
    (fmu_dir / "modelDescription.xml").write_text("<fmiModelDescription/>")
    (fmu_dir / "binaries" / "linux64" / "model.so").write_bytes(b"\x7fELF")
    return fmu_dir


class TestReadRootSystemName:
    def test_reads_name(self, tmp_path: Path):
        ssd = tmp_path / "SystemStructure.ssd"
        ssd.write_text(
            '<?xml version="1.0"?>'
            '<ssd:SystemStructureDescription xmlns:ssd="http://ssp-standard.org/SSP1"'
            ' name="doc" version="1.0">'
            '<ssd:System name="mysystem"/>'
            "</ssd:SystemStructureDescription>"
        )
        assert read_root_system_name(ssd) == "mysystem"

    def test_invalid_xml_returns_none(self, tmp_path: Path):
        ssd = tmp_path / "broken.ssd"
        ssd.write_text("<not xml")
        assert read_root_system_name(ssd) is None


class TestRepackagedFmus:
    def test_repackage_creates_fmu_archive(self, tmp_path: Path):
        fmu_dir = make_fmu_dir(tmp_path)
        packaged = repackage_unpacked_fmus(tmp_path)
        assert packaged == [tmp_path / "TestFMU.fmu"]
        assert not fmu_dir.exists()
        with zipfile.ZipFile(packaged[0]) as archive:
            names = set(archive.namelist())
        assert "modelDescription.xml" in names
        assert "binaries/linux64/model.so" in names

    def test_nested_fmus_packaged_innermost_first(self, tmp_path: Path):
        outer = make_fmu_dir(tmp_path, name="Outer")
        make_fmu_dir(outer, name="Inner")
        packaged = repackage_unpacked_fmus(tmp_path)
        assert tmp_path / "Outer.fmu" in packaged
        # The inner FMU is packaged first and ends up as an .fmu archive
        # inside the outer archive.
        with zipfile.ZipFile(tmp_path / "Outer.fmu") as archive:
            names = archive.namelist()
        assert "modelDescription.xml" in names
        assert "Inner.fmu" in names

    def test_package_fmu_dir_is_deterministic(self, tmp_path: Path):
        fmu_dir = make_fmu_dir(tmp_path)
        first = tmp_path / "first.fmu"
        second = tmp_path / "second.fmu"
        package_fmu_dir(fmu_dir, first)
        package_fmu_dir(fmu_dir, second)
        with zipfile.ZipFile(first) as a, zipfile.ZipFile(second) as b:
            assert a.namelist() == b.namelist()


class TestFindSsd:
    def test_prefers_default_name(self, tmp_path: Path):
        (tmp_path / "SystemStructure.ssd").write_text("<a/>")
        (tmp_path / "variant.ssd").write_text("<b/>")
        assert find_ssd(tmp_path).name == "SystemStructure.ssd"

    def test_falls_back_to_first_ssd(self, tmp_path: Path):
        (tmp_path / "variant.ssd").write_text("<b/>")
        assert find_ssd(tmp_path).name == "variant.ssd"

    def test_returns_none_without_ssd(self, tmp_path: Path):
        assert find_ssd(tmp_path) is None


class TestPrepareRuntimeSsp:
    def test_archive_input_is_usable_as_is(self, ssp_archive: Path):
        with prepare_runtime_ssp(ssp_archive) as runtime:
            assert zipfile.is_zipfile(runtime.archive)
            assert runtime.ssd_name == "SystemStructure.ssd"
            assert runtime.root_system == "system"
            with zipfile.ZipFile(runtime.archive) as archive:
                assert "SystemStructure.ssd" in archive.namelist()
                assert "resources/BouncingBall.fmu" in archive.namelist()

    def test_directory_input_repackages_unpacked_fmus(self, ssp_dir: Path):
        with prepare_runtime_ssp(ssp_dir) as runtime:
            with zipfile.ZipFile(runtime.archive) as archive:
                names = archive.namelist()
            assert "resources/BouncingBall.fmu" in names
            assert not any(
                name == "resources/BouncingBall/modelDescription.xml" for name in names
            )
            assert runtime.root_system == "system"

    def test_original_input_is_not_modified(self, ssp_dir: Path):
        before = sorted(p.relative_to(ssp_dir) for p in ssp_dir.rglob("*"))
        with prepare_runtime_ssp(ssp_dir):
            pass
        after = sorted(p.relative_to(ssp_dir) for p in ssp_dir.rglob("*"))
        assert before == after

    def test_missing_input_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            with prepare_runtime_ssp(tmp_path / "nope.ssp"):
                pass

    def test_directory_without_ssd_raises(self, tmp_path: Path):
        (tmp_path / "readme.txt").write_text("hi")
        with pytest.raises(ValueError, match="No .ssd file"):
            with prepare_runtime_ssp(tmp_path):
                pass
