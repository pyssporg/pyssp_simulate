"""Tests for the pyssp-simulate command line interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyssp_simulate.cli import main
from pyssp_simulate.result import SimulationResult, array_from_columns


def test_backends_command(capsys):
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    for name in ("fmpy", "omsimulator", "ssp4sim", "cosim"):
        assert name in out


def test_run_command(ssp_archive: Path, tmp_path: Path, spec_kwargs, capsys):
    output = tmp_path / "result.csv"
    exit_code = main(
        [
            "run",
            str(ssp_archive),
            "--backend",
            "fmpy",
            "--start-time",
            "0",
            "--stop-time",
            "0.2",
            "--step-size",
            "0.01",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert output.is_file()
    out = capsys.readouterr().out
    assert "backend=fmpy" in out


def test_run_command_unknown_backend(ssp_archive: Path, capsys):
    exit_code = main(["run", str(ssp_archive), "--backend", "nope"])
    assert exit_code == 1
    assert "Unknown backend" in capsys.readouterr().err


def test_compare_command(tmp_path: Path, capsys):
    def write(path: Path, offset: float) -> Path:
        time = np.linspace(0.0, 1.0, 11)
        data = array_from_columns(
            {"time": time, "Ball.h": 1.0 - time + offset}
        )
        SimulationResult(data=data, backend=path.stem).write_csv(path)
        return path

    first = write(tmp_path / "a.csv", 0.0)
    second = write(tmp_path / "b.csv", 1e-9)
    metrics = tmp_path / "metrics.csv"

    exit_code = main(["compare", str(first), str(second), "--atol", "1e-6",
                      "--output", str(metrics)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert metrics.is_file()


def test_compare_command_failing_tolerance(tmp_path: Path, capsys):
    def write(path: Path, values) -> Path:
        time = np.linspace(0.0, 1.0, 11)
        data = array_from_columns({"time": time, "Ball.h": values})
        SimulationResult(data=data, backend=path.stem).write_csv(path)
        return path

    first = write(tmp_path / "a.csv", np.zeros(11))
    second = write(tmp_path / "b.csv", np.ones(11))

    exit_code = main(["compare", str(first), str(second), "--atol", "1e-6"])
    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().out


def test_compare_command_needs_two_files(tmp_path: Path, capsys):
    exit_code = main(["compare", str(tmp_path / "only.csv")])
    assert exit_code == 2


def test_compare_command_with_reference_csv(tmp_path: Path, capsys):
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "time,default.Ball.time,default.Ball.h\n"
        "0.0,0.0,1.0\n"
        "1.0,1.0,0.5\n"
    )
    result = tmp_path / "fmpy.csv"
    time = np.linspace(0.0, 1.0, 3)
    data = array_from_columns({"time": time, "h": [1.0, 0.75, 0.55]})
    SimulationResult(data=data, backend="fmpy").write_csv(result)

    exit_code = main(
        [
            "compare",
            str(result),
            "--reference",
            str(reference),
            "--model-name",
            "Ball",
            "--start-time",
            "0",
            "--stop-time",
            "1",
            "--step-size",
            "0.5",
            "--atol",
            "0.1",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "reference vs fmpy" in out
    assert "PASS" in out
