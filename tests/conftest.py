"""Shared fixtures: a small SSP package around the Reference-FMUs BouncingBall.

The FMU (FMI 2.0, co-simulation capable, 2-clause BSD-licensed, from
https://github.com/modelica/Reference-FMUs) simulates a ball dropped
from 1 m with g = 9.81 m/s². Up to t = 0.4 s (before the first bounce
at ~0.45 s) the analytic solution is:

    h(t) = 1 - 0.5 * g * t²          (h(0.4) ≈ 0.2152)
    v(t) = -g * t                    (v(0.4) ≈ -3.924)

which gives the integration tests an engine-independent reference.

The fixture SSP is built with ``pyssp_standard`` (a dev dependency);
tests that only exercise pyssp_simulate itself do not require it.

The SSD carries a default experiment with an OSP fixed-step-algorithm
annotation that bakes in ``STEP_SIZE``. Backends with a step-size
command-line option pass their own value and ignore the annotation,
while the cosim backend (libcosim) has no such option and takes the
communication step from exactly this annotation.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
FMU_PATH = DATA_DIR / "BouncingBall.fmu"

# Time window that stays below the first bounce.
START_TIME = 0.0
STOP_TIME = 0.4

# Communication step size requested from every backend.
STEP_SIZE = 0.001


def analytic_height(t: float, g: float = 9.81) -> float:
    return 1.0 - 0.5 * g * t * t


@pytest.fixture(scope="session")
def bouncing_ball_fmu() -> Path:
    pytest.importorskip("pyssp_standard")
    assert FMU_PATH.is_file(), f"missing fixture FMU: {FMU_PATH}"
    return FMU_PATH


def _default_experiment():
    """Build the SSD default experiment with a fixed-step algorithm.

    libcosim (the cosim backend) reads the co-simulation algorithm from
    a ``com.opensimulationplatform`` annotation on the default
    experiment: an ``osp:FixedStepAlgorithm`` with a ``baseStepSize``.
    Without it cosim has no algorithm at all and crashes. Backends that
    take the step size from the command line simply ignore it.
    """
    from lxml import etree as ET
    from pyssp_standard import Annotation
    from pyssp_standard.ssd import DefaultExperiment
    from pyssp_standard.standard import ModelicaStandard

    osp_ns = "http://opensimulationplatform.com/SSP/OSPAnnotations"
    ssc_ns = ModelicaStandard.namespaces["ssc"]

    # libcosim matches the raw prefix-qualified tag names, so the "osp"
    # prefix must be declared exactly as written. It has to live on the
    # annotation element rather than on "osp:Algorithm" itself, which
    # libcosim would otherwise reject as an unknown algorithm.
    annotation_element = ET.Element(
        ET.QName(ssc_ns, "Annotation"),
        type="com.opensimulationplatform",
        nsmap={"osp": osp_ns},
    )
    algorithm = ET.SubElement(annotation_element, ET.QName(osp_ns, "Algorithm"))
    ET.SubElement(
        algorithm, ET.QName(osp_ns, "FixedStepAlgorithm"), baseStepSize=str(STEP_SIZE)
    )

    experiment = DefaultExperiment(None)
    experiment.start_time = START_TIME
    experiment.stop_time = STOP_TIME
    experiment.annotations.add_annotation(Annotation(annotation_element))
    return experiment


def _build_ssp(ssp_path: Path, fmu_path: Path) -> None:
    """Build a one-component SSP package around an FMU."""
    from pyssp_standard import SSP
    from pyssp_standard.ssd import Component, Connector, System

    with SSP(ssp_path, mode="w") as ssp:
        with ssp.system_structure as ssd:
            component = Component(None)
            component.component_type = "application/x-fmu-sharedlibrary"
            component.name = "BouncingBall"
            component.source = "resources/BouncingBall.fmu"
            component.implementation = "CoSimulation"
            component.connectors = [
                Connector(None, "h", "output"),
                Connector(None, "v", "output"),
            ]

            system = System(None, "system")
            system.elements.append(component)

            ssd.system = system
            ssd.name = "BouncingBall"
            ssd.version = "1.0"
            ssd.default_experiment = _default_experiment()

        ssp.add_resource(fmu_path)


@pytest.fixture(scope="session")
def ssp_archive(tmp_path_factory: pytest.TempPathFactory, bouncing_ball_fmu: Path) -> Path:
    """A packaged ``BouncingBall.ssp`` archive."""
    ssp_path = tmp_path_factory.mktemp("ssp_archive") / "BouncingBall.ssp"
    _build_ssp(ssp_path, bouncing_ball_fmu)
    assert zipfile.is_zipfile(ssp_path)
    return ssp_path


@pytest.fixture(scope="session")
def ssp_dir(tmp_path_factory: pytest.TempPathFactory, bouncing_ball_fmu: Path) -> Path:
    """An unpacked SSP tree whose FMU is also unpacked (a directory).

    Backends must repackage the FMU directory into an ``.fmu`` archive
    before simulating.
    """
    base = tmp_path_factory.mktemp("ssp_dir")
    archive_path = base / "BouncingBall.ssp"
    _build_ssp(archive_path, bouncing_ball_fmu)

    root = base / "BouncingBall"
    root.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)

    # Unpack the packaged FMU and remove the .fmu so only the
    # directory remains.
    fmu_dir = root / "resources" / "BouncingBall"
    with zipfile.ZipFile(root / "resources" / "BouncingBall.fmu") as archive:
        archive.extractall(fmu_dir)
    (root / "resources" / "BouncingBall.fmu").unlink()
    archive_path.unlink()
    return root


@pytest.fixture()
def spec_kwargs():
    """Common simulation settings for the fixture package."""
    return {
        "start_time": START_TIME,
        "stop_time": STOP_TIME,
        "step_size": STEP_SIZE,
    }
