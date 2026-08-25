# Text2Model Forge

Text2Model Forge is a local, auditable control plane for turning a text brief
into reviewed 2D and 3D asset candidates. It coordinates interchangeable model
workers, preserves every artifact and content hash, and stops at explicit human
gates before downstream work can continue.

This repository is a **developer preview**. Its deterministic coordinator,
contracts, browser UI, retries, invalidation rules, packaging, and lineage
checks are extensively tested. A complete live D0–D10 generation run has not
yet been independently reproduced, so this is not currently a dependable
one-click asset generator. Read the [cold public-readiness
assessment](docs/public-readiness.md) and [support matrix](SUPPORT.md) before
installing large model stacks.

## What it provides

- A typed eleven-stage text-to-model pipeline with resumable runs.
- Text-to-2D choices through Qwen Image, Z-Image Turbo, SDXL, or an existing
  ComfyUI service.
- Pluggable image-to-3D, Blender, retopology, motion, sprite, and engine workers.
- Browser controls for profiles, model backends, sequential candidate budgets,
  GPU policy, retries, corrections, cancellation, work-unit/VRAM progress,
  evidence inspection, and rollback.
- AI review recommendations that a person must explicitly confirm. The stored
  human decision remains append-only and is bound to both the review ID and the
  evidence hashes it saw.
- Fail-closed model lineage and export policy checks.
- Native Windows, Linux, and macOS setup/start scripts plus a non-root Docker
  deployment.

It does not train a foundation model, hide third-party model terms, guarantee
production-quality output, or make the unauthenticated Studio safe to expose to
the public internet.

## Pipeline

| Stage | Purpose | Human gate |
|---|---|---|
| D0 | Compile the brief into a typed asset contract | No |
| D1 | Generate and compare 2D concepts | Yes |
| D2 | Generate initial 3D geometry from the approved concept | No |
| D3 | Repair and retopologize geometry | No |
| D4 | Establish canonical landmarks and structure | Yes |
| D5 | Build articulation and rig | No |
| D6 | Validate skinning and deformation | No |
| D7 | Generate or retarget motion | Yes |
| D8 | Build surface materials and textures | Yes |
| D9 | Render directional sprites and LOD outputs | No |
| D10 | Validate the delivery in an isolated runtime project | Yes |

The D0 contract marks inapplicable stages as skipped. A static prop therefore
does not have to pass through rigging or motion. Reject, retry, edit, and
rollback invalidate affected downstream evidence; prior decisions are never
deleted.

## Start on a new machine

Clone the repository, then run it. `run.ps1`/`run.sh` is the one-command
entry point: it installs the control plane on first use (fast -- no GPU, no
model downloads) and just starts Studio on every run after that.

```powershell
.\run.ps1          # Windows
```

```bash
./run.sh           # Linux or macOS
```

Run it again any time; it is idempotent. Anything after the command is
forwarded straight through to the full launcher below, so `.\run.ps1 doctor`
or `./run.sh install --ai-stack qwen --accept-sdxl-license` both work.

The core install opens a guided offline walkthrough from the first-run page.
It creates a completed sample project with deterministic local PNG and GLB
artifacts, explicit synthetic-evidence labels, recorded gates, and provenance.
Use it to learn the interface before downloading a model stack. It is not a
generation result or qualification evidence.

For every install option -- the full local AI stack, repair, a deep doctor
report -- use the native launcher directly. `start` is idempotent: it installs
or repairs missing components and then starts Studio. Both launchers use
bounded retries, prefer non-admin installation, request elevation only when
needed, preserve an existing ignored `config.local.toml`, run a deterministic
smoke test, and show step progress.

Windows PowerShell:

```powershell
.\text2model-forge.ps1

# Explicit examples
.\text2model-forge.ps1 -Action install -AiStack core -IncludeDev
.\text2model-forge.ps1 -Action repair -AiStack existing -MaxAttempts 5
.\text2model-forge.ps1 -Action doctor -NoElevation -NoBrowser
```

Linux or macOS:

```bash
bash ./text2model-forge.sh

# Explicit examples
bash ./text2model-forge.sh install --ai-stack core --include-dev
bash ./text2model-forge.sh repair --ai-stack existing --max-attempts 5
bash ./text2model-forge.sh doctor --no-elevation --no-browser
```

Available AI stacks:

| Choice | Installs or uses |
|---|---|
| `auto` | Interactive selection, or safe automatic detection |
| `qwen` | Ollama reviewer, ComfyUI, Qwen Image, SDXL surface support, optional image-to-3D |
| `sdxl` | Ollama reviewer, ComfyUI, SDXL concept/surface support, optional image-to-3D |
| `existing` | Existing local ComfyUI, reviewer, Blender, and model installations |
| `core` | Control plane only; no large generation models |

Model downloads require their explicit license flags in non-interactive mode.
Weights and managed tools go under ignored `runtime/`; runs default to
`~/Text2ModelForgeRuns` (or the Windows user-profile equivalent).

For the measured low-VRAM path and its limitations, see the
[experimental 8 GB local stack](docs/8gb-local-stack.md).

The 8 GB profile does not pretend a small card is a 24 GB card. It serializes
GPU work, unloads the inactive model family, checks live free VRAM before each
heavy call, reserves driver/display headroom, uses tiled VAE decode, and spends
extra time on independent concept candidates. Deterministic pixel gates discard
unsafe candidates before the reviewer compares only the best few. This keeps
peak residency bounded; it improves the chance of a good input but cannot make
a small model mathematically equivalent to a 27B model or qualify untested 3D
quality.

## Docker

The container runs Studio as an unprivileged UID, persists work in a named
volume, and publishes Studio only on host loopback. Ollama is included;
GPU-heavy ComfyUI normally remains host-native.

```bash
docker compose config --quiet
docker compose up --build
```

Open <http://127.0.0.1:8766>. For an NVIDIA runtime, layer the GPU override:

```bash
docker compose -f compose.yaml -f compose.nvidia.yaml up --build
```

The default start does not download the reviewer model. Add `--profile models`
when you intentionally want Compose to pull the configured Ollama model.

Do not publish Studio directly to a LAN or the internet. It has no user account
or authorization boundary.

## Manual core installation

Python 3.12 is the supported runtime.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-dev.lock
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e .
.venv\Scripts\python -m text2model_forge doctor
.venv\Scripts\python -m text2model_forge demo --workspace C:/Text2ModelForgeRuns/demo
.venv\Scripts\python -m text2model_forge studio --workspace C:/Text2ModelForgeRuns --open-browser
```

On Linux or macOS replace `.venv\Scripts\python` with `.venv/bin/python`.
The lock files are hash-pinned exports from `pyproject.toml`; see
[Packaging](docs/packaging.md) before changing dependencies.

## Studio and headless control

Studio binds to `127.0.0.1` by default. Its primary UI is organized around
projects, six user-facing production phases, large evidence previews, and one
focused human-decision panel. The exact D0 to D10 stage record, generation
settings, worker controls, metrics, and append-only event history remain under
progressively disclosed technical views. The responsive layout reflows to a
single column at phone widths without hiding gate decisions or evidence.

A new project begins with a plain-language brief plus optional intended-use and
must-have fields. Model, profile, checkpoint, candidate-budget, and device
controls remain available under generation options. The 8 GB profile is
selected automatically when the detected primary GPU is in that class; all
profiles remain manually selectable.

```powershell
python -m text2model_forge studio --workspace C:/Text2ModelForgeRuns --open-browser
python -m text2model_forge studio list --workspace C:/Text2ModelForgeRuns
python -m text2model_forge studio show --workspace C:/Text2ModelForgeRuns --run-id <run-id>
python -m text2model_forge studio decide --workspace C:/Text2ModelForgeRuns `
  --run-id <run-id> --stage-id D1 --decision approve `
  --selected-evidence-id <evidence-id>
```

At a gate, a person may approve, reject, retry, edit with parameter overrides,
skip an inapplicable stage, or roll back. The AI recommendation has its own
confirmation button and is never submitted automatically. Headless clients can
record the same provenance with `--assisted-by-review-id <review-id>`.

Read-only versioned endpoints are available at `/api/v1/projects`,
`/api/v1/projects/<run-id>`, `/api/v1/projects/<run-id>/events`, and
`/api/v1/system/setup-options`. Browser and CLI processes serialize workspace
mutations through the same advisory lock, and stale revisions are rejected
instead of silently replacing newer run state.

## Configuration and workers

Configuration resolves in this order:

1. `src/text2model_forge/profiles/base.toml`
2. a named profile such as `simple`, `advanced`, or `8gb`
3. machine-local `config.local.toml`
4. per-run Studio overrides

Copy `machine.example.toml` to `config.local.toml` and edit only real local
paths. The file is ignored deliberately.

```powershell
python -m text2model_forge config show --profile simple
python -m text2model_forge config show --profile advanced
python -m text2model_forge workers
python -m text2model_forge doctor --deep
```

Worker manifests live in `resources/workers/`; measured model and hardware
evidence lives in `resources/qualifications/`. Lifecycle labels in those files,
not README prose, are authoritative. See the [worker guide](docs/worker-guide.md).

The optional directional-sprite subsystem is exposed as
`text2model-sprites`; see [Sprite renderer](docs/sprite-renderer.md).

## Repository layout

```text
src/text2model_forge/       Installable compiler, Studio, contracts, and sprites
resources/adapters/         Typed subprocess entry points for external workers
resources/workers/          Portable worker manifests
resources/qualifications/   Hash-bound worker/model qualification evidence
resources/blender/          Blender-side scripts
resources/unity_smoke_template/
                            Isolated optional runtime-validation project
resources/{assets,characters,creatures,genesis,presets}/
                            Portable examples and reusable contracts
docs/                       Maintained operator and architecture guides
tests/                      Deterministic unit, contract, CLI, and HTTP tests
docker/                     Container-only machine configuration
```

`src/` contains importable code. `resources/` contains files invoked or loaded
at runtime from either a source checkout or an installed wheel. Model weights,
generated runs, caches, secrets, and machine-specific configuration do not
belong in Git.

## Verification

The always-runnable checks require no GPU, Blender, ComfyUI, Ollama, or Unity:

```powershell
python -m pytest tests -q
python -m text2model_forge demo --workspace <temporary-directory>
python -m build
```

Tests use randomized order. A fake-worker pass proves coordinator behavior and
contracts; it does not qualify live model quality. Any worker change must state
which real hardware path, if any, was exercised.

## Documentation

- [Experimental 8 GB local stack](docs/8gb-local-stack.md)
- [Worker guide](docs/worker-guide.md)
- [Sprite renderer](docs/sprite-renderer.md)
- [Review rubric](docs/review-rubric.md)
- [Packaging and releases](docs/packaging.md)
- [Public-readiness assessment](docs/public-readiness.md)
- [Roadmap](ROADMAP.md), [support matrix](SUPPORT.md), and [security policy](SECURITY.md)

## Contributing and license

Contributions must preserve typed contracts, append-only decisions, artifact
hashing, lineage enforcement, and honest capability labels. See
[CONTRIBUTING.md](CONTRIBUTING.md).

The repository is licensed under Apache-2.0. Models, weights, source assets, and
generated outputs may have separate terms; review
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) before use or redistribution.
