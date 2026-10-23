"""Tests for the pyssp-simulate command line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyssp_simulate.cli import main


def test_backends_command(capsys):
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    for name in ("fmpy", "omsimulator", "ssp4sim", "cosim"):
        assert name in out


def test_run_command(ssp_archive: Path, tmp_path: Path, capsys):
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
    assert "backend=fmpy" in capsys.readouterr().out


def test_run_command_unknown_backend(ssp_archive: Path, capsys):
    """``--backend`` is restricted to the registered names by argparse."""
    with pytest.raises(SystemExit) as error:
        main(["run", str(ssp_archive), "--backend", "nope"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_compare_subcommand_is_gone():
    """Comparison moved to pyssp_vv; argparse must reject the old command."""
    with pytest.raises(SystemExit):
        main(["compare", "a.csv", "b.csv"])
