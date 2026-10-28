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

Clone the repository, then use the root launcher for your platform. It installs
the control plane on first use (fast -- no GPU, no
model downloads) and just starts Studio on every run after that.

```powershell
.\run.bat          # Windows
```

```bash
./run.command      # macOS
./run.sh           # Linux
```

PowerShell users can run `.\run.ps1` directly.

Run it again any time; it is idempotent. Anything after the command is
forwarded straight through to the full launcher below, so `.\run.ps1 doctor`
or `./run.sh install --ai-stack qwen --accept-sdxl-license` both work.

The root launchers share one vocabulary: `run` (the default), `doctor`,
`repair`, `docker`, `logs` and `stop`. On first run Studio opens a guided
offline walkthrough with a synthetic sample project, so you can learn the
interface before downloading any model stack.

Every install option, the AI stacks and their license flags, Docker, manual
installation, Studio and headless control, configuration and verification are
in [docs/setup.md](docs/setup.md).

## Documentation

- [Setup and operation](docs/setup.md)
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
