# Changelog

This project follows semantic versioning after `0.2.0`. During the preview
period, minor versions may change experimental worker and UI contracts.

## [Unreleased]

### Security

- Moved Pillow to 13.0 (`Pillow>=13.0,<14`; 12.3.0 has nine published advisories fixed in 13.0.0) and certifi to
  2026.10.19. The Security workflow's `dependency-audit` job was failing on these. The hash-checked
  `requirements*.lock` exports were regenerated with the pinned uv and now match `uv.lock`; they had fallen behind
  it (numpy 2.5.4, pydantic 2.14.1, trimesh 5.1.1 had been locked but never exported). The 448-test suite passes
  on this lock and `pip-audit` is clean.

### Added

- Public-release security, contribution, support, licensing, and release
  infrastructure.
- Static-prop golden-corpus specification and evidence contract.
- Cross-platform PowerShell and Bash setup/start/repair/doctor launchers.
- Docker Compose control-plane deployment.
- Hardware preflight, 8 GB experimental profile, headless Studio decisions,
  model/backend options, progress reporting, cancellation, and run controls.
- Expanded deterministic tests for configuration, Studio, launchers, hardware,
  and state transitions.
- Typed GPU-memory envelopes, live admission/peak telemetry, fail-closed device
  policies, sequential concept search, deterministic D1 quality gates, and
  tiled VAE decoding.

### Changed

- The public identity now makes the text-to-model scope explicit and avoids
  established product and Python-distribution collisions.
- Product claims now distinguish deterministic orchestration tests from live
  model qualification.
- The small-model D0 classifier now defines the complete asset-kind taxonomy,
  including the standalone-prop versus multi-object-environment boundary.
- Rigid-articulated stages now require motion explicitly requested by the
  brief; removable real-world parts no longer invent an unwanted rig.
- The managed NVIDIA ComfyUI launch reserves VRAM while keeping learned
  compute on GPU; the 8 GB profile uses SDXL by default and spends additional
  time on six sequential candidates rather than increasing peak residency.

## [0.1.0] - 2026-07-13

- Initial extracted D0–D10 compiler, worker protocol, lineage enforcement,
  qualification records, legacy sprite renderer, and deterministic demo.
