# pyssp_simulate

Simulate [SSP](https://ssp-standard.org) (System Structure and Parameterization)
packages through a single interface, with pluggable simulation engines.
Measurement and judgment of the results live in
[pyssp_vv](https://github.com/pyssporg/pyssp_vv); this package runs the
simulations and normalises what they produce.

pyssp_simulate is a sibling of pyssp_standard in the
[pyssporg](https://github.com/pyssporg) organization.

Supported backends:

| Name          | Engine                                                                                       | Install                                   |
| ------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `fmpy`        | [FMPy](https://pypi.org/project/FMPy/), pure-Python co-simulation master                     | PyPI, included in the base install        |
| `omsimulator` | [OMSimulator](https://pypi.org/project/OMSimulator/), OpenModelica's FMI co-simulation environment | PyPI, optional extra (see caveats)   |
| `ssp4sim`     | [ssp4sim](https://github.com/jkCXf9X4/ssp4sim), C++ engine with Python bindings              | GitHub releases, Linux x86_64, experimental |
| `cosim`       | [cosim](https://github.com/open-simulation-platform/cosim-cli), Open Simulation Platform command-line co-simulator | GitHub releases binary (experimental) |

## Installation

Requires Python 3.10+.

pyssp_simulate is not yet published on PyPI; install the wheel from a
[GitHub release](https://github.com/pyssporg/pyssp_simulate/releases) — no
checkout or git clone needed:

```bash
pip install https://github.com/pyssporg/pyssp_simulate/releases/download/v0.1.0/pyssp_simulate-0.1.0-py3-none-any.whl
```

(Bump the version in the URL to pin a different release.) Alternatively,
install straight from the repository:

```bash
pip install git+https://github.com/pyssporg/pyssp_simulate.git
```

or from a checkout:

```bash
pip install .
```

The base install includes FMPy, which installs reliably from PyPI on every
platform, so the default backend works out of the box. FMPy's SSP support is
intentionally simple: it records the values of the connectors declared in the
SSD, simulates FMUs with a fixed-step Jacobi scheme, and validates SSD files
against its bundled SSP 1.0 schema. Use the other backends for advanced SSP
features.

`simulate(spec, backend="auto")` (the default) uses the first available backend
in registry order: `fmpy`, `omsimulator`, `ssp4sim`, `cosim`.

Optional extras use the same syntax — `pip install ".[all]"` from a
checkout, `pip install "pyssp_simulate[all] @ <release-wheel-url>"` from a
release, or plain `pip install "pyssp_simulate[all]"` once the package is
published on PyPI:

```bash
pip install ".[all]"                  # every backend that is pip-installable
```

### OMSimulator

```bash
pip install ".[omsimulator]"
```

The OMSimulator package on PyPI is an sdist that downloads engine binaries
during installation, and the download URLs point at nightly build artifacts
that are eventually removed from the server. Installation therefore works on
current platforms but can fail on some distributions (for example Ubuntu
22.04). On affected systems, install OMSimulator manually (verified to work):

1. Download the nightly tarball for your platform from
   `https://build.openmodelica.org/omsimulator/nightly/<platform>/`
   (for example `linux-focal-amd64`).
2. Copy from the tarball into your environment's `site-packages`:
   - `lib/x86_64-linux-gnu/OMSimulator/` — the Python package directory,
   - `lib/x86_64-linux-gnu/libOMSimulator.so` — into the `OMSimulator` package
     directory.
3. Verify the install:

   ```bash
   python -c "from OMSimulator import SSP"
   ```

### ssp4sim (experimental)

ssp4sim is not on PyPI. Linux x86_64 wheels are published on
[GitHub releases](https://github.com/jkCXf9X4/ssp4sim/releases); the
known-good 0.2.11 build installs directly:

```bash
pip install https://github.com/jkCXf9X4/ssp4sim/releases/download/v0.2.11/pyssp4sim-0.2.11-cp39-abi3-linux_x86_64.whl
```

The engine is driven by a generated JSON configuration and runs in a
subprocess, because it does not reliably return control to the interpreter
after a simulation.

Wheel caveats (pyssp_simulate works around both):

- The wheels statically link libstdc++ but leave its symbols interposable.
  If a dynamic `libstdc++.so.6` is already loaded when `pyssp4sim` loads
  (typically via numpy → OpenBLAS → libgfortran), the process segfaults
  during startup. pyssp_simulate sidesteps this by running the engine in a
  child process that imports `pyssp4sim` before anything else.
- The v0.3.1 wheels contain AVX-512 instructions and crash with SIGILL on
  CPUs without AVX-512 support.

### cosim

[cosim](https://github.com/open-simulation-platform/cosim-cli) is the Open
Simulation Platform command-line co-simulator. It is not pip-installable:
download a release binary from
[cosim-cli releases](https://github.com/open-simulation-platform/cosim-cli/releases)
and put it on `PATH`, or point the `COSIM_BINARY` environment variable at it
(when set, `COSIM_BINARY` is used exclusively — a stale value disables the
backend rather than falling back to `PATH`).

cosim has no command-line step-size option: libcosim reads the communication
step from an OSP annotation in the SSD (a fixed-step-algorithm entry).
Packages without such an annotation fail in cosim itself — v0.8.0 exits with
a segfault — and `SimulationSpec.step_size` does not apply to this backend.

**Status: experimental.** The backend is validated against cosim v0.8.0 on
Linux x86_64 (the v0.9.0 release binaries require GLIBC ≥ 2.38); other
platforms and versions are untested.

### Why isn't everything in the base install?

FMPy is the only engine that installs reliably from PyPI on every platform,
so it is a core dependency. OMSimulator's sdist can fail on distributions whose
nightly build artifacts are no longer published, so it is opt-in. ssp4sim and
the cosim binary are not on PyPI at all.

## Quickstart

```python
from pyssp_simulate import SimulationSpec, simulate_all

spec = SimulationSpec(ssp="model.ssp", start_time=0.0, stop_time=1.0, step_size=0.001)
results = simulate_all(spec)          # every backend that is installed
for name, result in results.items():
    print(name, len(result.signals), len(result))
```

`simulate_all` runs every registered backend whose engine is installed and
emits a `RuntimeWarning` for each one it has to skip, so a base install (FMPy
only) yields a single result. Each result is a `SimulationResult` with a shared
time base and canonical `<component>.<signal>` names, ready to be handed to
`pyssp_vv.measure`.

To run a single backend:

```python
from pyssp_simulate import SimulationSpec, simulate

spec = SimulationSpec(ssp="model.ssp", start_time=0.0, stop_time=1.0, step_size=0.001)
result = simulate(spec, backend="fmpy")
```

## Results

Every backend returns a `SimulationResult` wrapping a structured NumPy array
with a `"time"` field plus one field per signal. Signal names are canonical
`<component>.<signal>` names — engine-specific path prefixes (such as
OMSimulator's root-system name) are stripped — so results from different
engines can be compared directly.

```python
result["Integrator.y"]   # one signal as a NumPy array
result.time                # the time grid
result.signals             # all signal names, "time" excluded
len(result)                # number of samples
```

Results round-trip through plain numeric CSVs:

```python
result.write_csv("fmpy.csv")
result = SimulationResult.from_csv("fmpy.csv")
```

`result.artifacts` collects files worth keeping. Its contents are
backend-dependent: every backend records the result CSV written through
`result_file` or `write_csv`, and backends that drive external engines also
record what they generated (for example ssp4sim's JSON config). The default
FMPy backend runs in-process and produces no extra artifacts.

## SimulationSpec

```python
from pyssp_simulate import SimulationSpec

spec = SimulationSpec(
    ssp="model.ssp",   # .ssp archive or unpacked SSP directory
    start_time=0.0,
    stop_time=10.0,
    step_size=None,     # None lets each backend pick its own default
    tolerance=None,     # optional solver/iteration tolerance
    result_file=None,   # write the result CSV here when set
    working_dir=None,   # backend scratch directory when set
    fixed_step=True,    # OMSimulator: also force setFixedStepSize(step)
    validate_ssd=True,  # FMPy: keep its bundled SSD schema validation
    omsimulator_options=(),  # OMSimulator engine command-line options
)
```

`ssp` accepts either an `.ssp` archive or an unpacked SSP directory (a
directory with a top-level `.ssd` file). Unpacked FMU directories — a directory
containing `modelDescription.xml` — are automatically repackaged into `.fmu`
archives at runtime. `prepare_runtime_ssp(source, repackage_fmus=False)` keeps
them as directories in both the runtime root and the archive, for engines that
load the unpacked layout natively (the ssp4sim backend does).

`tolerance` is honored by the OMSimulator and ssp4sim backends (ssp4sim
substitutes `1e-4` when left at `None`); FMPy and cosim have no tolerance
setting and ignore it.

`fixed_step` is an OMSimulator choice: with `False` the backend sets only the
logging interval and lets the master keep its event handling; with `True` it
also calls `setFixedStepSize`. `validate_ssd=False` skips FMPy's bundled SSP 1.0
schema validation for the runtime call. `omsimulator_options` are passed to
OMSimulator's `Capi.setCommandLineOption` before the package is loaded; an
option the engine rejects raises `RuntimeError`. Backends without the
corresponding choice ignore these fields.

## Measurement and judgment

Comparison is not part of this package. `pyssp_simulate` ends when a
`SimulationResult` exists.

| Question | Package |
| --- | --- |
| How far apart are two trajectories, under which alignment and normalisation? | `pyssp_vv.measure` |
| Is that acceptable, and why? | `pyssp_vv.judge` |

```python
from pyssp_vv.measure import Trajectory, compare

reference = Trajectory(a.time, {s: a[s] for s in a.signals}, label="fmpy")
candidate = Trajectory(b.time, {s: b[s] for s in b.signals}, label="omsimulator")
metrics = compare(reference, candidate)
```

How closely two engines agree is limited by the communication step size and
by how each master sequences component inputs and outputs — expect
differences on the order of the step size, growing with simulation length.
Use a small `step_size` when tight cross-engine agreement matters (the
quickstart above uses `0.001` for this reason).

## Permutation runs

A permutation config is a JSON file naming one SSP package and the sets of
simulation parameters to sweep. Each numeric field (`start_time`, `stop_time`,
`step_size`, `tolerance`) accepts a scalar, a list, or `null`, and the executed
runs are the cartesian product, each run with every configured backend:

```json
{
    "ssp": "model.ssp",
    "backends": ["fmpy", "omsimulator"],
    "stop_time": [1.0, 10.0],
    "step_size": [0.001, 0.01],
    "output_dir": "runs"
}
```

This describes 4 permutations (2 stop times × 2 step sizes) on two backends.
Only `ssp` is required, unknown keys are rejected (typo protection), and
backends whose engine is not installed are skipped with the reason recorded
on the run.

```python
from pyssp_simulate import load_permutation_config, run_permutations

runs = run_permutations(load_permutation_config("sweep.json"))
for run in runs:
    print(run.summary()["measurements"])  # metrics, or None without a second backend
```

With `output_dir` set, permutation `i` writes `runs/perm_<i>/<backend>.csv`
for every backend that ran, a `spec.json` manifest, and — when two or more
backends produced results — cross-backend metrics as
`runs/perm_<i>/metrics.csv`. Measuring requires `pyssp-vv[measure]`; without
it, permutations still run and the measurement is omitted.

## Backend registry

```python
from pyssp_simulate import available_backends, get_simulator

available_backends()   # {'fmpy': (True, None), 'omsimulator': (False, reason), ...}
get_simulator("fmpy")  # instantiate a backend by name
```

`available_backends()` reports each registered backend with the reason when it
is missing. `get_simulator(name)` raises `BackendNotAvailableError` with an
install hint when the backend's engine is not installed.

## Command line

The package installs a `pyssp-simulate` console script.

```console
$ pyssp-simulate run model.ssp --backend fmpy --stop-time 10 --step-size 0.001 --output result.csv
$ pyssp-simulate backends
$ pyssp-simulate run-all sweep.json
```

`run` simulates one SSP package and writes the result CSV given `--output`
(short form `-o`). It accepts `--backend` (`auto` by default, which picks the
first available backend), `--start-time`, `--stop-time` (default `1.0`),
`--step-size`, and `--tolerance`.

`backends` lists every registered backend and its availability.

`run-all` executes every permutation of a config file, printing one line per
run and a final summary. It reports metrics when two or more backends produced
a result; it never returns a pass/fail verdict. Measurement and judgment belong
to `pyssp_vv`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

Backends whose engines are not installed are skipped automatically.

The integration tests use two primitive FMUs — a ramp source (`y = t`) and
an integrator (`der(y) = u`) wired together — compiled at test time by the
[OpenModelica](https://openmodelica.org) compiler, so no prebuilt binaries
live in the repository. The analytic solutions (`t` and `t²/2`) give the
tests an engine-independent reference. The fixture SSP is assembled with
pyssp_standard (a dev dependency). If `omc` is not on your `PATH`, the
integration tests skip; to prepare the FMUs once and reuse them:

```bash
python tests/generate_fmus.py            # builds tests/fmus/build/*.fmu
export PYSSP_SIMULATE_TEST_FMU_DIR=tests/fmus/build
```

CI installs the OpenModelica compiler from the project's official apt
repository before running the tests.

## License

MIT — see [LICENSE](LICENSE).
