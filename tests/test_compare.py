"""Tests for result comparison on synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyssp_simulate.compare import compare, compare_pair, resample
from pyssp_simulate.result import SimulationResult, array_from_columns


def make_result(backend: str, time, height, velocity=None) -> SimulationResult:
    columns = {"time": np.asarray(time, dtype=float), "Ball.h": np.asarray(height, dtype=float)}
    if velocity is not None:
        columns["Ball.v"] = np.asarray(velocity, dtype=float)
    return SimulationResult(data=array_from_columns(columns), backend=backend)


class TestResample:
    def test_identity_grid(self):
        values = resample(np.array([0.0, 1.0]), np.array([5.0, 7.0]), np.array([0.0, 1.0]))
        np.testing.assert_allclose(values, [5.0, 7.0])

    def test_linear_interpolation(self):
        values = resample(np.array([0.0, 1.0]), np.array([0.0, 10.0]), np.array([0.25]))
        assert values[0] == pytest.approx(2.5)

    def test_out_of_range_clamps(self):
        values = resample(np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([-1.0, 2.0]))
        np.testing.assert_allclose(values, [1.0, 2.0])

    def test_nan_samples_are_dropped(self):
        times = np.array([0.0, 0.5, 1.0])
        values = np.array([0.0, np.nan, 10.0])
        resampled = resample(times, values, np.array([0.5]))
        assert resampled[0] == pytest.approx(5.0)

    def test_single_sample_constant(self):
        values = resample(np.array([0.0]), np.array([3.0]), np.array([-1.0, 1.0]))
        np.testing.assert_allclose(values, [3.0, 3.0])

    def test_all_nan_gives_nan(self):
        values = resample(np.array([0.0]), np.array([np.nan]), np.array([0.0]))
        assert np.isnan(values[0])


class TestComparePair:
    def test_identical_results(self):
        reference = make_result("a", [0.0, 0.5, 1.0], [1.0, 0.8, 0.6])
        candidate = make_result("b", [0.0, 0.5, 1.0], [1.0, 0.8, 0.6])
        report = compare_pair(reference, candidate)
        assert report.comparisons[0].max_abs_error == 0.0
        assert report.passed(atol=0.0)

    def test_constant_offset(self):
        reference = make_result("a", [0.0, 1.0], [1.0, 1.0])
        candidate = make_result("b", [0.0, 1.0], [1.5, 1.5])
        report = compare_pair(reference, candidate)
        item = report.comparisons[0]
        assert item.max_abs_error == pytest.approx(0.5)
        assert item.mean_abs_error == pytest.approx(0.5)
        assert item.rmse == pytest.approx(0.5)
        assert not report.passed(atol=0.1)

    def test_different_time_grids_resampled(self):
        reference = make_result("a", [0.0, 1.0, 2.0], [0.0, 10.0, 20.0])
        candidate = make_result("b", [0.0, 2.0], [1.0, 21.0])
        report = compare_pair(reference, candidate)
        # candidate at t=1.0 interpolates to 11.0 -> error 1.0 everywhere
        assert report.comparisons[0].max_abs_error == pytest.approx(1.0)

    def test_signal_subset(self):
        reference = make_result("a", [0.0, 1.0], [1.0, 2.0], velocity=[0.0, 1.0])
        candidate = make_result("b", [0.0, 1.0], [1.0, 3.0], velocity=[0.0, 1.0])
        report = compare_pair(reference, candidate, signals=["Ball.v"])
        assert [item.signal for item in report.comparisons] == ["Ball.v"]
        assert report.comparisons[0].max_abs_error == 0.0

    def test_missing_signal_raises(self):
        reference = make_result("a", [0.0, 1.0], [1.0, 2.0])
        candidate = make_result("b", [0.0, 1.0], [1.0, 2.0])
        with pytest.raises(KeyError, match="missing from one of the results"):
            compare_pair(reference, candidate, signals=["Ball.nope"])

    def test_disjoint_signals_use_common_only(self):
        reference = make_result("a", [0.0, 1.0], [1.0, 2.0], velocity=[0.0, 0.0])
        candidate = make_result("b", [0.0, 1.0], [1.0, 2.0])
        report = compare_pair(reference, candidate)
        assert [item.signal for item in report.comparisons] == ["Ball.h"]


class TestCompare:
    def test_needs_two_results(self):
        result = make_result("a", [0.0, 1.0], [1.0, 2.0])
        with pytest.raises(ValueError, match="at least two"):
            compare([result])

    def test_all_results_against_reference(self):
        reference = make_result("ref", [0.0, 1.0], [1.0, 1.0])
        close = make_result("close", [0.0, 1.0], [1.0, 1.05])
        far = make_result("far", [0.0, 1.0], [1.0, 2.0])
        report = compare([reference, close, far])
        assert [pair.candidate_backend for pair in report.pairs] == ["close", "far"]
        assert report.pairs[0].max_abs_error == pytest.approx(0.05)
        assert report.pairs[1].max_abs_error == pytest.approx(1.0)
        assert report.summary()["max_abs_error"] == pytest.approx(1.0)
        assert report.common_signals == ("Ball.h",)

    def test_pass_with_tolerances(self):
        reference = make_result("ref", [0.0, 1.0], [100.0, 100.0])
        candidate = make_result("cand", [0.0, 1.0], [100.0, 100.5])
        report = compare([reference, candidate])
        assert report.passed(atol=1.0)
        assert not report.passed(atol=0.1)
        assert report.passed(rtol=0.01)

    def test_write_csv(self, tmp_path: Path):
        reference = make_result("ref", [0.0, 1.0], [1.0, 1.0])
        candidate = make_result("cand", [0.0, 1.0], [1.0, 1.2])
        report = compare([reference, candidate])
        path = report.write_csv(tmp_path / "metrics.csv")
        text = path.read_text().splitlines()
        assert text[0].startswith("signal,reference_backend,candidate_backend")
        assert "Ball.h" in text[1]
        assert "cand" in text[1]
