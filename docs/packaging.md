# Packaging and release contract

Text2Model Forge supports three distribution paths:

1. a full Git checkout, used by the cross-platform launchers;
2. the Python wheel, which includes the CLI, browser UI, worker manifests,
   adapters, Blender helpers, audited registry, and Unity smoke template;
3. the Docker image, which runs the control plane as an unprivileged user.

The wheel installs executable resources under the active environment's
`share/text2model-forge` directory. `text2model_forge.paths.py` selects the repository
tree in an editable checkout and that shared directory in a normal install.
`TEXT2MODEL_FORGE_RESOURCE_ROOT` is an explicit escape hatch for distributors that
relocate data files. Machine-local worker bindings are never packaged; point
`TEXT2MODEL_FORGE_CONFIG` at a private `config.local.toml` or keep one in the source
checkout/current working directory.

Live qualification also binds each newly created run to the exact Git commit.
A source checkout discovers that revision from its own `.git` directory. A
wheel or distributor build must set `TEXT2MODEL_FORGE_SOURCE_REVISION` to a full
40-character commit SHA; the release container bakes the GitHub commit into
both this variable and its OCI revision label. An absent or malformed revision
does not stop ordinary Studio use, but it makes the golden qualification fail
closed.

CI builds a wheel and source distribution, installs the wheel into a new
environment outside the repository, loads all 13 worker manifests, resolves a
profile, and completes the deterministic D0–D10 demo. It also builds the
container and parses the Compose model.

## Dependency locks

`pyproject.toml` defines supported ranges. `uv.lock` records one portable
resolution, and the committed hash-checked exports divide it by install mode:

| File | Contents |
|---|---|
| `requirements.lock` | Runtime only |
| `requirements-local-ai.lock` | Runtime plus model-download tooling |
| `requirements-dev.lock` | Runtime plus tests, build, and audit tooling |
| `requirements-all.lock` | Runtime plus both optional groups |

Refresh all five files together with the pinned resolver, review the diff, and
run the full verification suite:

```bash
python -m pip install "uv==0.11.32"
uv lock --python 3.12
scripts/export-locks.sh
```

CI's `locks` job runs the same script and fails if any export differs from
`uv.lock`. Dependabot only edits `uv.lock`, so the `Dependabot lock exports`
workflow regenerates the exports on its pull requests and re-runs CI and the
security audit on the updated branch.

The project itself is then installed with `--no-deps --no-build-isolation` so
pip cannot silently resolve around the reviewed lock. ComfyUI and PyTorch stay
in ComfyUI's isolated environment and are not covered by these locks.

Pushing a signed release tag matching `v*` runs the full deterministic suite,
builds source and Python distributions, generates an SPDX JSON SBOM and
SHA-256 checksum manifest, records GitHub artifact attestations, and creates a
GitHub prerelease. This automation does not publish to PyPI and does not claim
that GPU workers passed live qualification.

Release tags must match `pyproject.toml` after normalizing Python's `rc` form:
for example, project version `0.2.0rc1` uses tag `v0.2.0-rc.1`.
