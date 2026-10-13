# pyssp_simulate

Simulate [SSP](https://ssp-standard.org) (System Structure and Parameterization)
packages through a single interface, with pluggable simulation engines, and
compare the results across engines.

pyssp_simulate is a sibling of pyssp_standard in the
[pyssporg](https://github.com/pyssporg) organization.

Supported backends:

| Name          | Engine                                                                                       | Install                                   |
| ------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `fmpy`        | [FMPy](https://pypi.org/project/FMPy/), pure-Python co-simulation master                     | PyPI, included in the base install        |
| `omsimulator` | [OMSimulator](https://pypi.org/project/OMSimulator/), OpenModelica's FMI co-simulation environment | PyPI, optional extra (see caveats)   |
| `ssp4sim`     | [ssp4sim](https://github.com/jkCXf9X4/ssp4sim), C++ engine with Python bindings              | GitHub releases, Linux x86_64, experimental |
| `cosim`       | [cosim](https://github.com/open-simulation-platform/cosim-cli), Open Simulation Platform command-line co-simulator | GitHub releases binary (experimental, future work) |

## Installation

Requires Python 3.10+.

```bash
pip install pyssp_simulate
```

The base install includes FMPy, which installs reliably from PyPI on every
platform, so the default backend works out of the box. FMPy's SSP support is
intentionally simple: it records the values of the connectors declared in the
SSD, simulates FMUs with a fixed-step Jacobi scheme, and validates SSD files
against its bundled SSP 1.0 schema. Use the other backends for advanced SSP
features.

`simulate(spec, backend="auto")` (the default) uses the first available backend
in registry order: `fmpy`, `omsimulator`, `ssp4sim`, `cosim`.

```bash
pip install "pyssp_simulate[all]"   # every backend that is pip-installable
```

### OMSimulator

```bash
pip install "pyssp_simulate[omsimulator]"
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
and put it on `PATH`, or point the `COSIM_BINARY` environment variable at it.

Note that cosim has no step-size flag; the communication step is taken from
the SSD's simulation information, so `SimulationSpec.step_size` does not apply to
this backend.

**Status: experimental.** The backend is validated against cosim v0.8.0 on
Linux (the v0.9.0 release binaries require GLIBC ≥ 2.38). libcosim needs an
OSP fixed-step-algorithm annotation in the SSD to select its master step;
system structures without such an annotation are rejected by cosim itself.

### Why isn't everything in the base install?

FMPy is the only engine that installs reliably from PyPI on every platform,
so it is a core dependency. OMSimulator's sdist can fail on distributions whose
nightly build artifacts are no longer published, so it is opt-in. ssp4sim and
the cosim binary are not on PyPI at all.

## Quickstart

```python
from pyssp_simulate import SimulationSpec, simulate_all, compare

spec = SimulationSpec(ssp="model.ssp", start_time=0.0, stop_time=10.0)
results = simulate_all(spec)          # every backend that is installed
report = compare(results)             # first result is the reference
print(report.summary())
```

To run a single backend:

```python
from pyssp_simulate import SimulationSpec, simulate

spec = SimulationSpec(ssp="model.ssp", start_time=0.0, stop_time=10.0)
result = simulate(spec, backend="fmpy")
```

## Results

Every backend returns a `SimulationResult` wrapping a structured NumPy array
with a `"time"` field plus one field per signal. Signal names are canonical
`<component>.<signal>` names — engine-specific path prefixes such as
OMSimulator's `model.root.` are stripped — so results from different engines
can be compared directly.

```python
result["BouncingBall.h"]   # one signal as a NumPy array
result.time                # the time grid
result.signals             # all signal names, "time" excluded
len(result)                # number of samples
```

Results round-trip through plain numeric CSVs:

```python
result.write_csv("fmpy.csv")
result = SimulationResult.from_csv("fmpy.csv")
```

`result.artifacts` collects files worth keeping (generated configs, logs, raw
engine output).

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
)
```

`ssp` accepts either an `.ssp` archive or an unpacked SSP directory (a
directory containing an `.ssd` file). Unpacked FMU directories — a directory
containing `modelDescription.xml` — are automatically repackaged into `.fmu`
archives at runtime.

## Comparing results

`compare` measures every result against the **first** one, which acts as the
reference. Candidate signals are resampled onto the reference time grid by
linear interpolation, then per-signal error metrics are computed:
`max_abs_error`, `mean_abs_error`, `rmse`, and the reference and candidate
min/max.

```python
report.passed(atol=1e-3, rtol=1e-6)  # every signal within tolerance?
report.write_csv("metrics.csv")      # per-signal metrics of every pair
report.summary()                     # dict summary for logging/JSON
```

A signal passes when its `max_abs_error` is at most
`atol + rtol * max(|reference_min|, |reference_max|)`.

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
$ pyssp-simulate compare fmpy.csv omsimulator.csv --atol 1e-3 --output metrics.csv
```

`run` simulates one SSP package (with `--backend auto` by default) and writes
the result CSV given `--output`. `backends` lists every registered backend and
its availability. `compare` takes two or more result CSVs, using the first as
the reference, and exits with code 0 when all signals pass the tolerance and 1
otherwise; `--json` prints the summary as JSON.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

Backends whose engines are not installed are skipped automatically. The
integration tests build an SSP package around the BouncingBall FMU from the
[Reference-FMUs](https://github.com/modelica/Reference-FMUs) project;
`tests/data/BouncingBall.fmu` is redistributed under that project's license
(the 2-clause BSD license, see
[LICENSE.txt](https://github.com/modelica/Reference-FMUs/blob/main/LICENSE.txt)).
The fixture package is assembled with pyssp_standard (a dev dependency), and
its analytic free-fall solution gives the tests an engine-independent
reference.

## License

MIT — see [LICENSE](LICENSE).
