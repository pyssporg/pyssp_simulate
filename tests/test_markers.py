"""Static checks on the collected markers and their selection.

Every test carries exactly one ``level_*`` marker (applied by file in
``tests/conftest.py`` unless the test overrides it), and the engine-running
set is selectable with ``-m level_integration``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.conftest import FILE_LEVELS, LEVEL_MARKERS

REPO_ROOT = Path(__file__).resolve().parents[1]

# Test functions known to run a real engine over the fixture SSP. Parameter
# sets (one per available backend) collapse onto the function name.
ENGINE_TEST_FUNCTIONS = {
    "test_fmpy_backend",
    "test_fmpy_backend_from_unpacked_dir",
    "test_fmpy_backend_writes_result_file",
    "test_omsimulator_backend",
    "test_ssp4sim_backend",
    "test_cosim_backend",
    "test_simulate_all_runs_available_backends",
    "test_cross_backend_agreement",
    "test_common_columns_have_identical_order",
    "test_omsimulator_real_backend_matches_recorded_header",
}


def _level_markers(item) -> list[str]:
    return [name for name in LEVEL_MARKERS if item.get_closest_marker(name)]


def test_every_collected_test_has_exactly_one_level_marker(collected_items):
    assert collected_items, "the collection hook recorded no tests"

    offenders = {
        item.nodeid: _level_markers(item)
        for item in collected_items
        if len(_level_markers(item)) != 1
    }

    assert offenders == {}, offenders


def test_level_markers_stay_within_the_file_budget(collected_items):
    """A file's tests carry its level; only test_backends overrides to contract."""
    offenders = []
    for item in collected_items:
        expected = FILE_LEVELS[item.path.name]
        actual = _level_markers(item)
        if actual == [expected]:
            continue
        if item.path.name == "test_backends.py" and actual == ["level_contract"]:
            continue
        offenders.append((item.nodeid, expected, actual))

    assert offenders == [], offenders


def test_slow_marker_is_opt_in_and_never_duplicated(collected_items):
    for item in collected_items:
        assert len(list(item.iter_markers("slow"))) <= 1, item.nodeid


def _collect(expression: str) -> set[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            expression,
            "tests/test_backends.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return {line.strip() for line in result.stdout.splitlines() if "::" in line}


def test_integration_marker_selects_the_engine_running_set():
    selected = _collect("level_integration")

    assert all(node.startswith("tests/test_backends.py::") for node in selected)
    selected_functions = {node.split("::")[1].split("[")[0] for node in selected}
    assert selected_functions == ENGINE_TEST_FUNCTIONS
    # The annotated contract tests in the same file are not engine runs.
    assert "test_fmpy_duplicate_canonical_columns_raise" not in selected_functions


def test_contract_marker_selects_no_engine_running_test():
    contract = _collect("level_contract")
    integration = _collect("level_integration")

    assert contract
    assert not contract & integration
