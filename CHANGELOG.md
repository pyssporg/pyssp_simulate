# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-11-24

### Added

- `replay_ssp4sim_config` replays an existing ssp4sim JSON configuration
  unchanged and normalises the result it wrote.
- Public ssp4sim runtime controls on `SimulationSpec`: `realtime`,
  `recording_interval`, and `log_fmu`.

### Changed

- Read engine result CSVs with `non_numeric="drop"`, so result columns that
  are not numeric no longer reach the normalised result.
- Use `pyssp_standard` for SSD and archive handling instead of the previous
  local implementation.
- Require Python 3.11 or newer.
- Bump the `pyssp-vv` pin to v0.3.0 and the `pyssp4sim` engine pin to v0.3.6.

## [0.3.0] - 2026-11-09

### Added

- Override a pip-installed engine with an environment variable pointing at a
  directory that contains the engine's Python package:
  `OMSIMULATOR_PYTHON_PATH` (omsimulator backend) and `SSP4SIM_PYTHON_PATH`
  (ssp4sim backend). A set but invalid override disables the backend instead of
  silently falling back to the pip-installed engine.

### Changed

- Bump the `pyssp-vv` pin to v0.2.0 and the `pyssp4sim` engine pin to v0.3.4.
- Re-pin `experimental-pyssp-standard` to the v0.9.3.dev0 release wheel.

