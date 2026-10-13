"""Tests for permutation configs, the batch runner and the ``run-all`` CLI.

The tests avoid asserting specific signal names: they check structural
properties only (counts, files, exit codes, dict keys), so they stay
independent of how the fixture package names its connectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyssp_simulate import available_backends
from pyssp_simulate.cli import main
from pyssp_simulate.permutations import (
    ComparisonSettings,
    PermutationConfig,
    iter_permutation_specs,
    load_permutation_config,
    run_permutations,
)

#: Tolerances used by the gated cross-backend comparison.
COMPARE_ATOL = 1e-2


def _backend_ready(name: str) -> bool:
    ok, _ = available_backends()[name]
    return ok


def _write_config(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


def _fake_ssp(tmp_path: Path) -> Path:
    """A placeholder file: SimulationSpec only checks that ``ssp`` exists."""
    ssp = tmp_path / "model.ssp"
    ssp.write_bytes(b"placeholder")
    return ssp


# --- config loading -------------------------------------------------------


def test_load_minimal_config_uses_defaults(tmp_path: Path, ssp_archive: Path):
    config_path = _write_config(tmp_path / "minimal.json", {"ssp": str(ssp_archive)})
    config = load_permutation_config(config_path)
    assert config.ssp == ssp_archive
    assert config.backends is None
    assert config.start_times == (0.0,)
    assert config.stop_times == (1.0,)
    assert config.step_sizes == (None,)
    assert config.tolerances == (None,)
    assert config.comparison == ComparisonSettings()
    assert config.output_dir is None


def test_load_normalizes_scalars_lists_and_nulls(tmp_path: Path):
    config_path = _write_config(
        tmp_path / "full.json",
        {
            "ssp": "some/model.ssp",
            "backends": ["FMPy", "OMSimulator"],
            "start_time": 0.5,
            "stop_time": [0.05, 0.1],
            "step_size": [0.001, None],
            "tolerance": None,
            "compare": {"atol": 1e-3, "rtol": 1e-5},
            "output_dir": "results",
        },
    )
    config = load_permutation_config(config_path)
    assert config.ssp == Path("some/model.ssp")
    assert config.backends == ("fmpy", "omsimulator")
    assert config.start_times == (0.5,)
    assert config.stop_times == (0.05, 0.1)
    assert config.step_sizes == (0.001, None)
    assert config.tolerances == (None,)
    assert config.comparison == ComparisonSettings(atol=1e-3, rtol=1e-5)
    assert config.output_dir == Path("results")


def test_load_rejects_unknown_keys(tmp_path: Path, ssp_archive: Path):
    config_path = _write_config(
        tmp_path / "typo.json", {"ssp": str(ssp_archive), "stop_tme": 0.1}
    )
    with pytest.raises(ValueError, match="stop_tme"):
        load_permutation_config(config_path)


def test_load_requires_ssp_key(tmp_path: Path):
    config_path = _write_config(tmp_path / "no_ssp.json", {"stop_time": 0.1})
    with pytest.raises(ValueError, match="ssp"):
        load_permutation_config(config_path)


# --- spec generation ------------------------------------------------------


def test_iter_permutation_specs_is_the_cartesian_product(tmp_path: Path):
    config = PermutationConfig(
        ssp=_fake_ssp(tmp_path),
        backends=("fmpy",),
        stop_times=(0.1, 0.2),
        step_sizes=(0.001, None),
    )
    specs = list(iter_permutation_specs(config))
    assert len(specs) == 4
    assert {(spec.stop_time, spec.step_size) for spec in specs} == {
        (0.1, 0.001),
        (0.1, None),
        (0.2, 0.001),
        (0.2, None),
    }
    assert all(spec.ssp == config.ssp for spec in specs)
    assert all(spec.start_time == 0.0 for spec in specs)
    assert all(spec.tolerance is None for spec in specs)


def test_empty_sweep_yields_no_runs(tmp_path: Path):
    config = PermutationConfig(
        ssp=_fake_ssp(tmp_path), backends=("fmpy",), stop_times=()
    )
    assert list(iter_permutation_specs(config)) == []
    assert run_permutations(config) == []


def test_run_permutations_records_skipped_backends(tmp_path: Path, monkeypatch):
    """Unavailable backends are recorded instead of raising."""
    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    monkeypatch.setattr(
        FMPySimulator,
        "available",
        classmethod(lambda cls: (False, "engine not installed")),
    )
    output_dir = tmp_path / "results"
    config = PermutationConfig(
        ssp=_fake_ssp(tmp_path),
        backends=("fmpy",),
        stop_times=(0.05,),
        output_dir=output_dir,
    )
    runs = run_permutations(config)
    assert len(runs) == 1
    run = runs[0]
    assert run.results == {}
    assert run.skipped == {"fmpy": "engine not installed"}
    assert run.report is None
    assert run.passed is None
    assert run.summary()["skipped"] == {"fmpy": "engine not installed"}
    # The manifest is still written even when no backend ran.
    assert (run.output_dir / "spec.json").is_file()
    assert not (run.output_dir / "fmpy.csv").exists()


# --- run_permutations -----------------------------------------------------


def test_run_permutations_writes_artifacts(tmp_path: Path, ssp_archive: Path):
    output_dir = tmp_path / "results"
    config_path = _write_config(
        tmp_path / "config.json",
        {
            "ssp": str(ssp_archive),
            "backends": ["fmpy"],
            "stop_time": [0.05, 0.1],
            "step_size": 0.001,
            "output_dir": str(output_dir),
        },
    )
    runs = run_permutations(load_permutation_config(config_path))
    assert len(runs) == 2
    for index, run in enumerate(runs):
        run_dir = output_dir / f"perm_{index:04d}"
        assert run.output_dir == run_dir
        assert set(run.results) == {"fmpy"}
        assert run.skipped == {}
        assert len(run.results["fmpy"]) > 0
        assert run.report is None
        assert run.passed is None
        assert (run_dir / "fmpy.csv").is_file()
        assert (run_dir / "spec.json").is_file()
        assert not (run_dir / "metrics.csv").exists()

        summary = run.summary()
        assert set(summary) == {
            "spec",
            "backends",
            "skipped",
            "report",
            "passed",
            "output_dir",
        }
        assert summary["backends"] == ["fmpy"]
        assert summary["skipped"] == {}
        assert summary["report"] is None
        assert summary["passed"] is None
        assert summary["output_dir"] == str(run_dir)
        assert summary["spec"] == json.loads((run_dir / "spec.json").read_text())


@pytest.mark.skipif(not _backend_ready("omsimulator"), reason="OMSimulator not installed")
def test_run_permutations_compares_backends(tmp_path: Path, ssp_archive: Path):
    config_path = _write_config(
        tmp_path / "config.json",
        {
            "ssp": str(ssp_archive),
            "backends": ["fmpy", "omsimulator"],
            "stop_time": 0.05,
            "step_size": 0.001,
            "compare": {"atol": COMPARE_ATOL},
            "output_dir": str(tmp_path / "results"),
        },
    )
    runs = run_permutations(load_permutation_config(config_path))
    assert len(runs) == 1
    run = runs[0]
    assert set(run.results) == {"fmpy", "omsimulator"}
    assert run.skipped == {}
    assert run.report is not None
    assert run.passed is True
    assert run.report.pairs[0].reference_backend == "fmpy"
    for name in run.results:
        assert (run.output_dir / f"{name}.csv").is_file()
    assert (run.output_dir / "metrics.csv").is_file()
    assert (run.output_dir / "spec.json").is_file()
    report_summary = run.report.summary()
    assert report_summary["backends"] == ["fmpy", "omsimulator"]
    assert report_summary["pairs"]


# --- CLI -------------------------------------------------------------------


def test_cli_run_all_single_backend(tmp_path: Path, ssp_archive: Path, capsys):
    config_path = _write_config(
        tmp_path / "config.json",
        {
            "ssp": str(ssp_archive),
            "backends": ["fmpy"],
            "stop_time": 0.05,
            "step_size": 0.001,
        },
    )
    assert main(["run-all", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "perm_0000" in out
    assert "fmpy" in out
    assert "total permutations=1" in out
    assert "failed=0" in out


def test_cli_run_all_zero_permutations(tmp_path: Path, ssp_archive: Path, capsys):
    config_path = _write_config(
        tmp_path / "config.json", {"ssp": str(ssp_archive), "stop_time": []}
    )
    assert main(["run-all", str(config_path)]) == 1
    assert "zero permutations" in capsys.readouterr().err


def test_cli_run_all_rejects_unknown_keys(tmp_path: Path, ssp_archive: Path, capsys):
    config_path = _write_config(
        tmp_path / "config.json", {"ssp": str(ssp_archive), "backedns": ["fmpy"]}
    )
    assert main(["run-all", str(config_path)]) == 1
    assert "backedns" in capsys.readouterr().err
