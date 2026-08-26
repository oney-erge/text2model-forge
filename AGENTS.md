# AGENTS.md

## Project

Text2Model Forge is a text-to-3D asset compiler: a description goes in, and a
human-gated animated (or static) asset comes out through eleven named stages,
D0 through D10. `src/text2model_forge/` is the compiler -- orchestration, contracts,
worker protocol, human gates, config resolution, and the browser control
plane. `src/text2model_forge/sprites/` is the older directional-sprite renderer text2model_forge's D8/D9
stages reuse for its ComfyUI multiview client and Blender view baker.

Nothing here may depend on a specific consuming project or character/creature
identity: this is a general-purpose tool. If you find a consumer-specific
reference, that is a regression; fix it at the root rather than adding a
special case.

Use these durable sources of truth:

- `README.md` for the pipeline overview, configuration, and human-control
  actions.
- `docs/worker-guide.md` for detailed worker-by-worker commands and
  qualification history.
- `docs/sprite-renderer.md` for the directional renderer's own commands.
- `src/text2model_forge/profiles/base.toml` for every configuration default and its
  meaning -- this file's comments are the parameter reference.
- `src/text2model_forge/schemas.py` and `src/text2model_forge/studio_models.py` for the data
  contracts every stage and gate decision must satisfy.

## Commands

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
python -m pytest tests -q
python -m text2model_forge demo --workspace C:/Text2ModelForgeRuns/demo
python -m text2model_forge config show --profile advanced
python -m text2model_forge studio --workspace C:/Text2ModelForgeRuns --open-browser
python -m text2model_forge studio list --workspace C:/Text2ModelForgeRuns
python -m text2model_forge doctor --deep
python -m text2model_forge workers
```

`demo` and the test suite need no GPU, Blender, ComfyUI, or Unity -- they are
the fast, always-runnable checks. `studio` and `workers` reflect live worker
availability from `config.local.toml`, which is machine-specific and
gitignored; copy `machine.example.toml` to get one.

## Architecture Boundaries

- `src/text2model_forge/schemas.py`, `src/text2model_forge/studio_models.py`: the typed contracts.
  Adding a field means updating the contract, not reading an untyped dict at
  the call site.
- `src/text2model_forge/compiler.py`, `src/text2model_forge/workers.py`, `src/text2model_forge/external_worker.py`:
  the generic job lifecycle (validate -> plan -> execute -> collect artifacts
  -> hard gates -> human approval -> promote). Stage-specific logic belongs in
  `src/text2model_forge/studio_pipeline.py`'s `_run_dN` methods, not here.
- `src/text2model_forge/lineage.py`, `src/text2model_forge/export_policy.py`: license/lineage
  enforcement. This is the most mature, most safety-critical part of the
  system -- it is what stops a gated or territory-restricted model from
  silently reaching a release export. Never weaken a check here to unblock a
  single asset; fix the asset's lineage instead.
- `src/text2model_forge/hardware.py`, `src/text2model_forge/preflight.py`: what this machine is,
  what stack that implies, and whether every cross-stage assumption holds.
  Every constant in `recommend_stack()` was discovered by a failed pipeline
  run; adding a new backend means adding its assumption here, not just its
  code, or the next person rediscovers it the same expensive way. A check
  that reports a problem must carry a `remedy` -- a diagnosis without a fix
  just moves the guessing.
- `src/text2model_forge/settings.py`: the layered configuration resolver. It only merges
  plain data and knows nothing about StudioRun; `studio_overrides()` is the
  one place that translates resolved config into StudioRun constructor
  fields. Adding a new tunable parameter means adding it to
  `src/text2model_forge/profiles/base.toml` with a comment explaining it, not inventing a
  new ad hoc default somewhere else.
- `resources/adapters/*.py`: one worker's subprocess entry point each, speaking the
  `ExternalWorkerRequest`/response file contract. An adapter must not import
  from `src/text2model_forge/studio_pipeline.py`; the dependency runs the other way.
- Repo-relative resource resolution (`Path(__file__).resolve().parents[N]`)
  is real and load-bearing throughout `resources/adapters/`, `src/text2model_forge/`, and
  `src/text2model_forge/sprites/` -- these packages assume they are running from within a full
  checkout (editable install), not as pure importable libraries. If you
  change a file's directory depth, grep for `parents[` and fix every
  offset; the test suite mostly (not entirely) catches this by exercising
  the affected code paths, so also check by hand.

## Human Gate Invariants

- A gate decision (`StudioStore.decide()`) always hash-binds the evidence it
  saw and appends to an append-only `human_decisions` list. Never make a
  decision mutate or delete history -- that traceability is the whole point
  of the human-review contract.
- `reject`, `retry`, and `edit` invalidate everything downstream of the
  stage they're recorded against; `skip` does not; `rollback` invalidates
  from its target stage forward, inclusive. Get this wrong and a run can
  promote work built on since-invalidated evidence.
- `_invalidate_from()` resets a stage to `pending` only when
  `stage.applicable` is True. A stage D0's compiled contract ruled out
  (`applicable=False` -- a static prop's rig, a material's geometry) stays
  `skipped` with its reason intact, because invalidating later work says
  nothing about whether the asset needs it. `_run_d0` is therefore the only
  place that may lift an exclusion, and it must reset applicability for
  every stage before recomputing it -- otherwise a re-compiled spec can
  never regain a stage the previous spec excluded.
- Rejection (and edit, when it carries no explicit comment) requires a
  comment -- so the record explains what should change, not just that
  something was wrong.
- Do not add a new decision type without updating `_DECISIONS`,
  `_invalidate_from`'s callers, the CLI/web layer, and a test that exercises
  its state-machine effect end to end.
- Never filter human decisions on a literal `decision == "reject"`. Use
  `CORRECTION_DECISIONS` / `latest_correction()` / `awaiting_correction()`
  from `src/text2model_forge/studio_models.py`, so `edit` keeps behaving like `reject`
  everywhere. Getting this wrong does not fail loudly -- it silently drops
  the human's correction and re-runs the stage as an unrelated fresh
  generation. There are eight such lookups across `studio_pipeline.py` and
  `studio_qwen.py`; all eight must agree.
- A parameter override parked on a stage by a retry/edit is consumed once by
  `_begin()`, which returns it and clears it. A stage runner that wants
  overrides must use `_begin()`'s return value; anything that stores an
  override without a reader is dead code.

## Change Style

- Inspect the relevant schema, config default, and existing test before
  editing.
- Make the smallest coherent change; reuse `src/text2model_forge/settings.py`,
  `src/text2model_forge/schemas.py`, and the existing worker protocol rather than adding
  a parallel mechanism.
- A new stage parameter belongs in `src/text2model_forge/profiles/base.toml`, documented,
  with a WIRED or DOCUMENTED marker in its section comment depending on
  whether `studio_pipeline.py` actually reads it yet.
- Keep `src/text2model_forge/` and `src/text2model_forge/sprites/` decoupled from any specific consuming
  project. Treat any consumer-specific identifier as a regression.
- Prefer extending an existing adapter's typed request/response contract
  over adding a new ad hoc script.

## Verification

- Any change to `src/text2model_forge/` or `src/text2model_forge/sprites/`: run `pytest tests -q` and
  `python -m text2model_forge demo --workspace <tmp>`. Both are fast, deterministic,
  and require no external tools -- there is no excuse to skip them.
- Test order is randomized (`pytest-randomly`). A failure prints its seed;
  reproduce with `--randomly-seed=N`. Do not "fix" an order-dependent
  failure by pinning the order -- it is a real shared-state bug. The one
  found so far was two `StudioStore` instances over one directory holding
  independent locks.
- A stage-level test drives the real coordinator with `worker_executor=`
  and/or `script_runner=` injected (see `tests/test_studio.py`). Fakes must
  emit the exact output roles the stage consumes and real parseable
  artifacts where the stage parses them -- a stub file that merely exists
  passes for the wrong reason and hides defects.
- A regression test must be proven to fail without its fix: revert, confirm
  the exact original error, restore. State that you did this.
- A change to the gate/decision state machine (`studio_store.py`,
  `studio_models.py`): add or update a test in `tests/test_studio.py` that
  exercises the state transition directly, following the existing
  `_awaiting_stage_with_one_candidate`-style pattern rather than driving the
  full coordinator when you only need to test `decide()`'s mechanics.
- A change to `src/text2model_forge/settings.py` or `profiles/*.toml`: add a case to
  `tests/test_settings.py` and confirm `text2model_forge config show` still resolves
  correctly for `simple` and `advanced`.
- A change to `src/text2model_forge/config.py` (worker bindings): re-verify
  `machine.example.toml` still round-trips through `load_local_config`.
- A change to an adapter, a Blender/ComfyUI script, or anything requiring a
  real worker: state plainly that it was not exercised end to end unless you
  actually ran it against real Blender/ComfyUI/GPU/Unity, and say so in the
  handoff. Do not claim a live-worker check ran without observed evidence.

## Git and Safety

- Preserve unrelated changes and keep commits focused.
- Use the configured repository-owner identity.
- Do not add assistant names, co-author trailers, session links, or tool
  attribution to Git artifacts.
- Never commit `config.local.toml` (real machine paths) -- it is gitignored;
  keep it that way.
- `resources/qualifications/*.json` records real hardware and licensing findings about
  third-party models (some gated, some territory-restricted). Read one before
  assuming it is safe to change or remove; the lineage/export-policy system
  depends on these records staying accurate.
- Finish with what changed, what was verified, what was skipped, and
  remaining risks -- especially anything that needed a live worker this
  environment did not have.


## Install and run contract

- Keep `run.bat`, `run.ps1`, `run.command`, and `run.sh` as the stable
  user entry points. They must keep the same `run`, `doctor`, `repair`,
  `docker`, `logs`, and `stop` actions where the application supports them.
- Use the `native-app-delivery` Codex skill when changing first-run setup,
  repair, Docker, or launcher behavior. That is an internal workflow name and
  must not appear in product copy or the public README.
- Keep shared install mechanics in `scripts/install-utils.ps1` and
  `scripts/install-utils.sh`. Preserve idempotent reruns, bounded transient
  retries, install locking, disk checks, user state, and `.setup/install.log`.
- Verify launcher changes with PowerShell parsing, `bash -n`, the focused
  delivery audit, and `docker compose config`. Do not run the full application
  test suite unless the change affects application behavior.
