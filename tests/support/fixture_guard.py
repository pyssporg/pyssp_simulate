"""Session guard: committed fixtures are read-only during a test run.

``tests/fixtures`` holds the reviewed recordings and ``tests/fmus`` the
committed Modelica sources; both are snapshotted at session start and
re-digested at the end. Any change sets the session exit status to failure and
names the paths, so a test that writes into a fixture fails the run instead of
silently updating the baseline. Generated FMUs live in a temporary directory
(or the ``PYSSP_SIMULATE_TEST_FMU_DIR`` cache) and are not guarded; the
``tests/fmus/build`` output directory is excluded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]

#: Roots that must not change during a run.
GUARDED_ROOTS = (_TESTS / "fixtures", _TESTS / "fmus")

#: Top-level names under a guarded root that are generated, not committed.
IGNORED_TOP_LEVEL = frozenset({"build"})

_STATE: dict[Path, dict[str, str]] = {}


def _digest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.relative_to(root).parts[0] not in IGNORED_TOP_LEVEL
    }


def snapshot() -> None:
    """Record the digest of every guarded root that exists."""
    for root in GUARDED_ROOTS:
        if root.is_dir():
            _STATE[root] = _digest(root)


def check(session) -> None:
    """Fail the session when a guarded root changed since :func:`snapshot`."""
    drifted = []
    for root, before in _STATE.items():
        after = _digest(root)
        drifted += [
            f"{root.relative_to(_TESTS).as_posix()}/{name}"
            for name in sorted(set(before) | set(after))
            if before.get(name) != after.get(name)
        ]
    if not drifted:
        return
    session.exitstatus = 1
    message = "committed fixtures changed during the run:\n" + "\n".join(
        f"  {path}" for path in drifted
    )
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(message, red=True)
    else:
        print(message)
