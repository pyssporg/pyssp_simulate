"""Tests for the normalized result container and signal canonicalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from pyssp_simulate.result import (
    SimulationResult,
    array_from_columns,
    canonicalize_signal,
    read_dymola_mat,
    read_result_csv,
    write_result_csv,
)


class TestCanonicalizeSignal:
    def test_time_is_case_insensitive(self):
        assert canonicalize_signal("time") == "time"
        assert canonicalize_signal("Time") == "time"
        assert canonicalize_signal(" TIME ") == "time"

    def test_plain_component_signal_untouched(self):
        assert canonicalize_signal("BouncingBall.h") == "BouncingBall.h"

    def test_single_segment_untouched(self):
        assert canonicalize_signal("h") == "h"

    def test_engine_root_prefixes_are_dropped(self):
        assert canonicalize_signal("root.BouncingBall.h") == "BouncingBall.h"
        assert canonicalize_signal("model.root.BouncingBall.h") == "BouncingBall.h"
        assert canonicalize_signal("fmu.BouncingBall.h") == "BouncingBall.h"
        assert canonicalize_signal("fmu.h") == "h"
        assert canonicalize_signal("system.fmu.h", "system") == "h"

    def test_root_system_name_is_dropped(self):
        assert canonicalize_signal("system.BouncingBall.h", "system") == "BouncingBall.h"

    def test_container_name_sequence_is_dropped(self):
        # OMSimulator 3.0.0.post198+ prefixes columns with the SSD name on
        # top of the system name: <ssd>.<system>.<component>.<signal>.
        containers = ("RampToIntegrator", "system")
        assert (
            canonicalize_signal("RampToIntegrator.system.RampSource.y", containers)
            == "RampSource.y"
        )
        # Older versions only prefix the system name; both must work with
        # the same container set.
        assert canonicalize_signal("system.RampSource.y", containers) == "RampSource.y"

    def test_container_names_never_eat_real_signals(self):
        # A component two segments deep is never stripped, even when its
        # name matches a container level.
        assert canonicalize_signal("system.h", ("system", "doc")) == "system.h"

    def test_two_segments_are_never_stripped(self):
        # "model.x" could be a genuine component named model.
        assert canonicalize_signal("model.h") == "model.h"
        assert canonicalize_signal("root.h", "root") == "root.h"

    def test_nested_systems_keep_their_path(self):
        assert canonicalize_signal("subsys.fmuA.signal") == "subsys.fmuA.signal"
        assert canonicalize_signal("model.subsys.fmuA.signal") == "subsys.fmuA.signal"

    def test_quotes_and_whitespace_are_stripped(self):
        assert canonicalize_signal(' "BouncingBall.h" ') == "BouncingBall.h"


class TestArrayFromColumns:
    def test_orders_fields(self):
        data = array_from_columns(
            {
                "time": np.array([0.0, 1.0]),
                "b": np.array([1.0, 2.0]),
                "a": np.array([3.0, 4.0]),
            }
        )
        assert data.dtype.names == ("time", "a", "b")
        assert len(data) == 2

    def test_order_is_independent_of_insertion_order(self):
        values = {
            "time": np.array([0.0, 1.0]),
            "b": np.array([1.0, 2.0]),
            "a": np.array([3.0, 4.0]),
        }
        first = array_from_columns(dict(values))
        second = array_from_columns(dict(reversed(list(values.items()))))
        assert first.dtype.names == second.dtype.names

    def test_empty_columns_rejected(self):
        with pytest.raises(ValueError):
            array_from_columns({})


class TestCsvRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        data = array_from_columns(
            {
                "time": np.linspace(0.0, 1.0, 11),
                "ball.h": np.linspace(1.0, 0.5, 11),
            }
        )
        path = write_result_csv(tmp_path / "result.csv", data)
        loaded = read_result_csv(path)
        assert loaded.dtype.names == data.dtype.names
        np.testing.assert_allclose(loaded["ball.h"], data["ball.h"])

    def test_read_canonicalizes_headers(self, tmp_path: Path):
        path = tmp_path / "result.csv"
        path.write_text("Time,root.Ball.h\n0.0,1.0\n0.5,0.8\n")
        loaded = read_result_csv(path, root_system="system")
        assert loaded.dtype.names == ("time", "Ball.h")
        assert loaded["Ball.h"][1] == pytest.approx(0.8)

    def test_read_empty_csv_raises(self, tmp_path: Path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_result_csv(path)

    def test_read_duplicate_columns_raise(self, tmp_path: Path):
        path = tmp_path / "dup.csv"
        path.write_text("a.b,a.b\n1.0,1.0\n")
        with pytest.raises(ValueError, match="Duplicate"):
            read_result_csv(path)


class TestSimulationResult:
    def make_result(self) -> SimulationResult:
        data = array_from_columns(
            {
                "time": np.array([0.0, 0.5, 1.0]),
                "Ball.h": np.array([1.0, 0.9, 0.6]),
                "Ball.v": np.array([0.0, -1.0, -2.0]),
            }
        )
        return SimulationResult(data=data, backend="fake")

    def test_signals_exclude_time(self):
        result = self.make_result()
        assert result.signals == ("Ball.h", "Ball.v")

    def test_getitem(self):
        result = self.make_result()
        np.testing.assert_allclose(result["Ball.h"], [1.0, 0.9, 0.6])
        np.testing.assert_allclose(result.time, [0.0, 0.5, 1.0])

    def test_getitem_unknown_signal_raises(self):
        result = self.make_result()
        with pytest.raises(KeyError, match="nope"):
            result["Ball.nope"]

    def test_write_csv_records_artifact(self, tmp_path: Path):
        result = self.make_result()
        path = result.write_csv(tmp_path / "out.csv")
        assert path in result.artifacts
        assert SimulationResult.from_csv(path).backend == "unknown"

    def test_summary(self):
        result = self.make_result()
        summary = result.summary()
        assert summary["backend"] == "fake"
        assert summary["signals"] == ["Ball.h", "Ball.v"]
        assert summary["samples"] == 3
        assert summary["time_range"] == [0.0, 1.0]


def _char_matrix(strings: list[str]) -> np.ndarray:
    """A Dymola/MATLAB-style character matrix (one string per column)."""
    width = max(len(text) for text in strings)
    return np.array(
        [[ord(character) for character in text.ljust(width)] for text in strings],
        dtype=np.uint16,
    ).T


def _write_mat(path: Path, *, keys: set[str] | None = None) -> Path:
    """Write a Dymola ``dsres`` MAT file with one of each data source."""
    matrices = {
        "Aclass": _char_matrix(["Trajectory"]),
        "name": _char_matrix(["time", "speed", "constant", "negated", "skipped"]),
        "description": _char_matrix(
            ["Independent time axis", "speed", "constant", "negated", "skipped"]
        ),
        "dataInfo": np.array(
            [
                [0, 2, 1, 2, 3],
                [0, 2, 1, -2, 1],
            ],
            dtype=np.int64,
        ),
        "data_1": np.array([[5.0]], dtype=np.float64),
        "data_2": np.array(
            [
                [0.0, 1.0, 2.0],
                [10.0, 20.0, 30.0],
            ],
            dtype=np.float64,
        ),
    }
    selected = {name: value for name, value in matrices.items() if keys is None or name in keys}
    savemat(path, selected)
    return path


class TestReadDymolaMat:
    def test_decodes_names_and_data_sources(self, tmp_path: Path):
        data = read_dymola_mat(_write_mat(tmp_path / "dsres.mat"))

        assert data.dtype.names == ("time", "constant", "negated", "speed")
        np.testing.assert_allclose(data["time"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(data["speed"], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(data["constant"], [5.0, 5.0, 5.0])
        np.testing.assert_allclose(data["negated"], [-10.0, -20.0, -30.0])

    def test_output_feeds_write_result_csv(self, tmp_path: Path):
        data = read_dymola_mat(_write_mat(tmp_path / "dsres.mat"))
        path = write_result_csv(tmp_path / "result.csv", data)
        reloaded = read_result_csv(path)

        assert reloaded.dtype.names == data.dtype.names
        np.testing.assert_allclose(reloaded["negated"], data["negated"])

    def test_missing_matrix_raises(self, tmp_path: Path):
        path = _write_mat(tmp_path / "dsres.mat", keys={"name", "data_2"})
        with pytest.raises(KeyError, match="Missing expected MAT keys"):
            read_dymola_mat(path)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="MAT file not found"):
            read_dymola_mat(tmp_path / "none.mat")

    def test_is_exported_from_the_package(self):
        import pyssp_simulate

        assert pyssp_simulate.read_dymola_mat is read_dymola_mat
