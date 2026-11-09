# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Override a pip-installed engine with an environment variable pointing at a
  directory that contains the engine's Python package:
  `OMSIMULATOR_PYTHON_PATH` (omsimulator backend) and `SSP4SIM_PYTHON_PATH`
  (ssp4sim backend). A set but invalid override disables the backend instead of
  silently falling back to the pip-installed engine.

