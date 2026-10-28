"""Static policy for committed fixtures: layout, provenance, and size.

Every committed fixture is a case directory under ``tests/fixtures`` with an
``input.<ext>`` and a ``case.md`` (``tests/fixtures/README.md`` explains the
rule). The policy is checked here so a new case cannot land as a loose file or
without provenance, and so no test reaches a fixture through the working
directory instead of ``tests.support.fixtures``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from tests.support.fixtures import FIXTURES

pytestmark = pytest.mark.level_static

TESTS = Path(__file__).resolve().parent

#: Required sections in every ``case.md``: the metadata the bytes cannot carry.
REQUIRED_HEADINGS = (
    "- **Requirement / fault:**",
    "- **Independent oracle:**",
    "- **Volatile fields:**",
    "- **Update policy:**",
    "## What this case proves",
    "## Update history",
)

#: Largest committed fixture file, in bytes, unless listed below.
MAX_FIXTURE_BYTES = 64 * 1024

#: Fixture path relative to ``tests/fixtures`` -> reason for exceeding the cap.
SIZE_ALLOWLIST: dict[str, str] = {}

#: Files allowed directly under ``tests/fixtures`` (the index only).
ROOT_FILES = {"README.md"}

#: Written from parts so this module does not itself contain the literal.
CWD_FIXTURE_NEEDLE = "tests" + "/fixtures"


def _case_directories() -> list[Path]:
    return sorted(path for path in FIXTURES.iterdir() if path.is_dir())


def test_every_fixture_directory_has_a_case_record():
    offenders = [
        path.name for path in _case_directories() if not (path / "case.md").is_file()
    ]

    assert offenders == [], f"fixture directories without case.md: {offenders}"


def test_no_loose_fixture_files_under_the_root():
    offenders = sorted(
        path.name
        for path in FIXTURES.iterdir()
        if path.is_file() and path.name not in ROOT_FILES
    )

    assert offenders == [], f"fixture files outside a case directory: {offenders}"


def test_every_case_has_an_input_file():
    offenders = [
        path.name
        for path in _case_directories()
        if not any(child.name.startswith("input.") for child in path.iterdir())
    ]

    assert offenders == [], f"fixture directories without input.*: {offenders}"


def test_every_case_record_states_the_required_provenance():
    offenders = {}
    for path in _case_directories():
        case = path / "case.md"
        if not case.is_file():
            continue
        text = case.read_text(encoding="utf-8")
        missing = [heading for heading in REQUIRED_HEADINGS if heading not in text]
        if missing:
            offenders[path.name] = missing

    assert offenders == {}, offenders


def test_no_test_source_uses_a_cwd_relative_fixture_path():
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.resolve() == Path(__file__).resolve():
            # This module names the policy it enforces; it is not a consumer.
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CWD_FIXTURE_NEEDLE in line or "Path.cwd()" in line:
                offenders.append(f"{path.relative_to(TESTS)}:{number}")

    assert offenders == [], f"fixture paths reached from the working directory: {offenders}"


def test_fixture_files_stay_within_the_size_budget():
    offenders = {}
    for path in sorted(FIXTURES.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(FIXTURES).as_posix()
        size = path.stat().st_size
        if size > MAX_FIXTURE_BYTES and relative not in SIZE_ALLOWLIST:
            offenders[relative] = size

    assert offenders == {}, f"fixture files over {MAX_FIXTURE_BYTES} bytes: {offenders}"


def test_fixture_allowlist_entries_name_real_files():
    missing = sorted(
        name for name in SIZE_ALLOWLIST if not (FIXTURES / name).is_file()
    )

    assert missing == [], missing


def test_case_records_use_the_expected_metadata_shape():
    """Metadata bullets are ``- **Label:** value``; a bare label fails."""
    pattern = re.compile(r"^- \*\*[^*]+\*\*\s+\S", re.MULTILINE)
    offenders = {}
    for path in _case_directories():
        case = path / "case.md"
        if not case.is_file():
            continue
        text = case.read_text(encoding="utf-8")
        labels = [
            line for line in text.splitlines() if line.startswith("- **")
        ]
        if not labels or not all(pattern.match(line) for line in labels):
            offenders[path.name] = labels

    assert offenders == {}, offenders
