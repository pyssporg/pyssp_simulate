"""Tests for SimulationSpec validation and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyssp_simulate.spec import SimulationSpec, spec_summary


def test_spec_accepts_archive(tmp_path: Path):
    ssp = tmp_path / "model.ssp"
    ssp.write_bytes(b"not really a zip, existence is all we check")
    spec = SimulationSpec(ssp=ssp, start_time=0.0, stop_time=1.0)
    assert spec.ssp == ssp
    assert spec.duration == 1.0


def test_spec_accepts_directory(tmp_path: Path):
    (tmp_path / "SystemStructure.ssd").write_text("<ssd/>")
    spec = SimulationSpec(ssp=tmp_path, start_time=1.0, stop_time=2.0)
    assert spec.duration == 1.0


def test_spec_rejects_nested_ssd_directory(tmp_path: Path):
    # An SSD below the top level is not a valid unpacked SSP package;
    # reject it at construction instead of failing at simulate time.
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "SystemStructure.ssd").write_text("<ssd/>")
    with pytest.raises(ValueError, match="top-level .ssd"):
        SimulationSpec(ssp=tmp_path, start_time=0.0, stop_time=1.0)


def test_spec_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SimulationSpec(ssp=tmp_path / "missing.ssp")


def test_spec_rejects_non_ssp_directory(tmp_path: Path):
    (tmp_path / "unrelated.txt").write_text("hello")
    with pytest.raises(ValueError, match="unpacked SSP"):
        SimulationSpec(ssp=tmp_path, start_time=0.0, stop_time=1.0)


def test_spec_rejects_bad_time_window(tmp_path: Path):
    ssp = tmp_path / "model.ssp"
    ssp.write_bytes(b"")
    with pytest.raises(ValueError, match="stop_time"):
        SimulationSpec(ssp=ssp, start_time=1.0, stop_time=1.0)


@pytest.mark.parametrize("kwargs", [{"step_size": 0.0}, {"step_size": -1.0}])
def test_spec_rejects_bad_step_size(tmp_path: Path, kwargs):
    ssp = tmp_path / "model.ssp"
    ssp.write_bytes(b"")
    with pytest.raises(ValueError, match="step_size"):
        SimulationSpec(ssp=ssp, start_time=0.0, stop_time=1.0, **kwargs)


def test_spec_rejects_bad_tolerance(tmp_path: Path):
    ssp = tmp_path / "model.ssp"
    ssp.write_bytes(b"")
    with pytest.raises(ValueError, match="tolerance"):
        SimulationSpec(ssp=ssp, start_time=0.0, stop_time=1.0, tolerance=0.0)


def test_effective_step_size_fallback(tmp_path: Path):
    ssp = tmp_path / "model.ssp"
    ssp.write_bytes(b"")
    spec = SimulationSpec(ssp=ssp, start_time=0.0, stop_time=10.0)
    assert spec.effective_step_size() == pytest.approx(0.1)

    spec = SimulationSpec(ssp=ssp, start_time=0.0, stop_time=10.0, step_size=0.5)
    assert spec.effective_step_size() == 0.5


def test_workspace_temporary_by_default(tmp_path: Path):
    ssp = tmp_path / "model.ssp"
    ssp.write_bytes(b"")
    spec = SimulationSpec(ssp=ssp, start_time=0.0, stop_time=1.0)

    with spec.workspace() as work:
        assert work.is_dir()
        marker = work / "marker.txt"
        marker.write_text("x")
    assert not work.exists()


def test_workspace_honors_working_dir(tmp_path: Path):
    ssp = tmp_path / "model.ssp"
    ssp.write_bytes(b"")
    target = tmp_path / "keep"
    spec = SimulationSpec(ssp=ssp, start_time=0.0, stop_time=1.0, working_dir=target)

    with spec.workspace() as work:
        assert work == target
        assert work.is_dir()
    assert target.exists()


def test_spec_summary_roundtrip(tmp_path: Path):
    ssp = tmp_path / "model.ssp"
    ssp.write_bytes(b"")
    spec = SimulationSpec(
        ssp=ssp, start_time=0.0, stop_time=2.0, step_size=0.1, tolerance=1e-4
    )
    summary = spec_summary(spec)
    assert summary["ssp"] == str(ssp)
    assert summary["stop_time"] == 2.0
    assert summary["step_size"] == 0.1
    assert summary["result_file"] is None
