"""Tests for the FMU fixture generator and its provenance manifest.

The manifest is what stops ``PYSSP_SIMULATE_TEST_FMU_DIR`` from silently
reusing FMUs that no longer correspond to the Modelica sources.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests import generate_fmus as generator


def _fake_fmus(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in generator.FMU_NAMES:
        (directory / name).write_bytes(f"fmu:{name}".encode())


def test_manifest_round_trip(tmp_path: Path):
    _fake_fmus(tmp_path)

    path = generator.write_manifest(tmp_path)

    assert path.is_file()
    assert generator.verify_manifest(tmp_path) == []


def test_manifest_detects_changed_fmu(tmp_path: Path):
    _fake_fmus(tmp_path)
    generator.write_manifest(tmp_path)
    (tmp_path / generator.FMU_NAMES[0]).write_bytes(b"rebuilt differently")

    problems = generator.verify_manifest(tmp_path)

    assert any("does not match its recorded digest" in problem for problem in problems)


def test_manifest_detects_missing_fmu(tmp_path: Path):
    _fake_fmus(tmp_path)
    generator.write_manifest(tmp_path)
    (tmp_path / generator.FMU_NAMES[0]).unlink()

    problems = generator.verify_manifest(tmp_path)

    assert any("is missing" in problem for problem in problems)


def test_manifest_requires_the_manifest_file(tmp_path: Path):
    _fake_fmus(tmp_path)

    problems = generator.verify_manifest(tmp_path)

    assert any(generator.MANIFEST_NAME in problem for problem in problems)


def test_manifest_detects_changed_model_sources(tmp_path: Path, monkeypatch):
    _fake_fmus(tmp_path)
    generator.write_manifest(tmp_path)
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in generator.SCRIPT_NAMES:
        (sources / name).write_text("changed model source")
    monkeypatch.setattr(generator, "FMU_SOURCE_DIR", sources)

    problems = generator.verify_manifest(tmp_path)

    assert any("different model sources" in problem for problem in problems)


def test_generate_rejects_a_stale_cache_before_compiling(tmp_path: Path):
    _fake_fmus(tmp_path)
    generator.write_manifest(tmp_path)
    (tmp_path / generator.FMU_NAMES[0]).write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="stale FMU cache"):
        generator.generate_fmus(tmp_path)
