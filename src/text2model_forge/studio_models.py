"""Versioned persistence contracts for the local Text2Model Studio control plane."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Literal

from pydantic import Field, model_validator
from text2model_forge.paths import source_revision

from .schemas import StrictModel


STAGE_DEFINITIONS = (
    ("D0", "Brief", False),
    ("D1", "Concept", True),
    ("D2", "3D generation", False),
    ("D3", "Cleanup", False),
    ("D4", "Canonical structure / skeleton", True),
    ("D5", "Rig / articulation", False),
    ("D6", "Skinning / deformation", False),
    ("D7", "Motion", True),
    ("D8", "Surface painting", True),
    ("D9", "Sprite rendering", False),
    ("D10", "Runtime validation", True),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StudioEquipment(StrictModel):
    equipment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    category: Literal["weapon", "shield", "armor", "attachment"]
    side: Literal["left", "right", "center"]
    socket: str = Field(min_length=1)
    grip: Literal["none", "palm_and_fingers", "forearm_strap"]
    rigid: bool = True
    description: str = Field(min_length=1)
    visual_requirements: list[str] = Field(default_factory=list)


class StudioComponent(StrictModel):
    """A named part whose identity, connection, and possible motion must survive production."""

    component_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    role: Literal["body", "movable_part", "attachment", "surface", "effect"]
    connection: str = Field(min_length=1)
    motion: Literal["none", "rigid", "deformable", "simulated"] = "none"
    description: str = Field(min_length=1)
    visual_requirements: list[str] = Field(default_factory=list)


class StudioAssetSpec(StrictModel):
    schema_version: Literal[1] = 1
    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    creative_direction: str = Field(min_length=1)
    asset_kind: Literal[
        "character", "creature", "prop", "architecture", "environment", "material", "vfx"
    ] = "character"
    behavior: Literal["static", "rigid_articulated", "deformable_animated", "simulated"] = (
        "deformable_animated"
    )
    anatomy_family: Literal["humanoid", "short_biped", "quadruped", "custom"] | None = "humanoid"
    height_m: float | None = Field(default=1.8, gt=0.01, lt=1000.0)
    dimensions_m: list[float] = Field(default_factory=lambda: [1.0, 1.8, 1.0], min_length=3, max_length=3)
    silhouette: list[str] = Field(min_length=1)
    materials: list[str] = Field(min_length=1)
    components: list[StudioComponent] = Field(default_factory=list)
    equipment: list[StudioEquipment] = Field(default_factory=list)
    animations: list[str] = Field(default_factory=list)
    locked_features: list[str] = Field(min_length=1)
    negative_constraints: list[str] = Field(min_length=1)
    gameplay_readability: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_asset_contract(self) -> "StudioAssetSpec":
        if any(value <= 0 for value in self.dimensions_m):
            raise ValueError("asset dimensions must be positive")
        for animation in self.animations:
            if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", animation):
                raise ValueError("animation ids must be lowercase and machine-safe")
        if self.behavior == "static" and self.animations:
            raise ValueError("static assets cannot declare animation clips")
        if self.asset_kind in {"character", "creature"} and self.behavior == "deformable_animated":
            if self.anatomy_family is None:
                raise ValueError("deformable characters and creatures require an anatomy family")
        return self


# Compatibility name retained for existing adapters and saved character runs.
StudioCharacterSpec = StudioAssetSpec


class StudioEvidence(StrictModel):
    evidence_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    label: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)


class StudioQwenReview(StrictModel):
    review_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    stage_id: str = Field(pattern=r"^D(?:10|[0-9])$")
    iteration: int = Field(ge=1)
    summary: str = Field(min_length=1)
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    candidate_ranking: list[str] = Field(default_factory=list)
    recommended_evidence_id: str | None = None
    recommended_changes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    hard_requirements_satisfied: bool = True
    request_human_review: bool = True


class StudioHumanDecision(StrictModel):
    decision_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    # approve/reject are the original two verdicts. retry re-runs the same
    # stage with no implied quality judgement (a reroll). edit is a reject
    # that carries a concrete correction in `overrides` for the next attempt.
    # skip marks a stage not applicable without invalidating anything after
    # it. rollback reopens an earlier, already-decided stage and invalidates
    # everything from there forward -- see StudioStore.decide().
    decision: Literal["approve", "reject", "retry", "edit", "skip", "rollback"]
    comment: str = ""
    selected_evidence_id: str | None = None
    evidence_hashes: dict[str, str] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    target_stage_id: str | None = None
    # Set only when a person explicitly confirms a recommendation rendered
    # from this stage attempt's AI review. The decision remains human-owned;
    # this field records the recommendation provenance without inventing a
    # second, unaudited gate path.
    assisted_by_review_id: str | None = None


# Decisions whose comment and overrides are a correction that the next attempt
# of the same stage must consume. "reject" is the original verdict; "edit" is
# the same thing with an explicit correction attached. Anything that reads "the
# human's latest correction" must treat both identically -- filtering on
# "reject" alone silently discards an edit's comment and overrides.
CORRECTION_DECISIONS = frozenset({"reject", "edit"})


def latest_correction(stage: "StudioStageState") -> "StudioHumanDecision | None":
    """The most recent decision carrying a correction for the next attempt."""
    return next(
        (
            item
            for item in reversed(stage.human_decisions)
            if item.decision in CORRECTION_DECISIONS
        ),
        None,
    )


def awaiting_correction(stage: "StudioStageState") -> bool:
    """True when this stage's most recent decision asked for another attempt
    with a correction, rather than approving, skipping, or plain-retrying."""
    return bool(
        stage.human_decisions and stage.human_decisions[-1].decision in CORRECTION_DECISIONS
    )


def validate_stage_overrides(overrides: dict[str, Any] | None) -> None:
    """Reject malformed per-attempt override values at decision time.

    Without this the check happens deep inside the stage runner, so a typo
    like {"seed": -5} is accepted by the gate, reported as success, and only
    surfaces later as an asynchronously failed stage -- far from the input
    that caused it. Validating here means the web form (and any API caller)
    gets an immediate, actionable error instead.

    Unknown keys are deliberately allowed: a stage may define its own
    overrides, and this must not become a chokepoint that has to be updated
    before any stage can add one. Only the shape of known keys is enforced.
    """
    if not overrides:
        return
    seed = overrides.get("seed")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
    ):
        raise ValueError("the 'seed' override must be a non-negative whole number")
    for key in ("concept_steps",):
        value = overrides.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 150
        ):
            raise ValueError(f"the '{key}' override must be a whole number between 1 and 150")
    concept_candidates = overrides.get("concept_candidates")
    if concept_candidates is not None and (
        not isinstance(concept_candidates, int)
        or isinstance(concept_candidates, bool)
        or not 2 <= concept_candidates <= 12
    ):
        raise ValueError("the 'concept_candidates' override must be a whole number between 2 and 12")
    for key in ("concept_width", "concept_height"):
        value = overrides.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or not 256 <= value <= 2048
        ):
            raise ValueError(f"the '{key}' override must be a whole number between 256 and 2048")
    for key in ("concept_cfg",):
        value = overrides.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value <= 30
        ):
            raise ValueError(f"the '{key}' override must be a number greater than 0 and at most 30")
    render_size = overrides.get("render_size")
    if render_size is not None and (
        not isinstance(render_size, int) or isinstance(render_size, bool) or not 64 <= render_size <= 4096
    ):
        raise ValueError("the 'render_size' override must be a whole number between 64 and 4096")
    for key in ("maximum_material_change_fraction",):
        value = overrides.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1
        ):
            raise ValueError(f"the '{key}' override must be a number greater than 0 and at most 1")
    bone_influences = overrides.get("maximum_bone_influences")
    if bone_influences is not None and (
        not isinstance(bone_influences, int) or isinstance(bone_influences, bool) or not 1 <= bone_influences <= 8
    ):
        raise ValueError("the 'maximum_bone_influences' override must be a whole number between 1 and 8")
    landmark_adjustments = overrides.get("landmark_adjustments")
    if landmark_adjustments is not None:
        # Only the top-level shape: resources/adapters/blender_worker.py owns the deep
        # semantics (landmark names, offset bounds) and rejects a malformed
        # value itself. Duplicating that here would drift out of sync with it.
        if not isinstance(landmark_adjustments, dict):
            raise ValueError("the 'landmark_adjustments' override must be an object")
        for name, offset in landmark_adjustments.items():
            if not isinstance(name, str) or not isinstance(offset, list) or len(offset) != 3:
                raise ValueError(
                    "each 'landmark_adjustments' entry must map a landmark name to a 3-value offset"
                )
            if not all(isinstance(component, (int, float)) and not isinstance(component, bool) for component in offset):
                raise ValueError("each 'landmark_adjustments' offset must contain three numbers")
    weight_adjustments = overrides.get("weight_adjustments")
    if weight_adjustments is not None:
        # Same boundary as landmark_adjustments above: WEIGHT_JOINT_PAIRS and
        # the transfer/radius fraction bounds live in the adapter, not here.
        if not isinstance(weight_adjustments, list) or not all(
            isinstance(item, dict) for item in weight_adjustments
        ):
            raise ValueError("the 'weight_adjustments' override must be an array of objects")


class StudioStageState(StrictModel):
    stage_id: str = Field(pattern=r"^D(?:10|[0-9])$")
    label: str = Field(min_length=1)
    gate_required: bool
    state: Literal[
        "pending",
        "queued",
        "running",
        "awaiting_review",
        "approved",
        "rejected",
        "blocked",
        "failed",
        "skipped",
    ] = "pending"
    applicable: bool = True
    progress: float = Field(default=0, ge=0, le=1)
    progress_phase: str = "waiting"
    progress_current: int = Field(default=0, ge=0)
    progress_total: int = Field(default=0, ge=0)
    progress_unit: str = ""
    gpu_used_gb: float | None = Field(default=None, ge=0)
    gpu_free_gb: float | None = Field(default=None, ge=0)
    gpu_total_gb: float | None = Field(default=None, ge=0)
    retry_attempt: int = Field(default=0, ge=0)
    retry_reason: str = ""
    message: str = "Waiting for the preceding stage."
    iteration: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    evidence: list[StudioEvidence] = Field(default_factory=list)
    metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    qwen_reviews: list[StudioQwenReview] = Field(default_factory=list)
    human_decisions: list[StudioHumanDecision] = Field(default_factory=list)
    error: str | None = None
    # Set by a retry or edit decision; consumed and cleared by the next run
    # of this stage. Empty for an ordinary approve/reject-driven attempt.
    pending_overrides: dict[str, Any] = Field(default_factory=dict)


class StudioRun(StrictModel):
    schema_version: Literal[1] = 1
    # Optimistic concurrency token maintained by StudioStore. Older run.json
    # files load as revision 0 and are upgraded on their next successful
    # write. The field is part of the persisted contract so a browser process
    # and a concurrent CLI process cannot silently overwrite each other.
    revision: int = Field(default=0, ge=0)
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    # Tutorial runs contain clearly-labelled deterministic sample artifacts.
    # They exercise the same browser, evidence, and audit surfaces but are
    # never presented as generated or qualification evidence.
    run_mode: Literal["production", "tutorial"] = "production"
    source_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    description: str = Field(min_length=1)
    intended_use: str | None = Field(default=None, max_length=120)
    must_have_features: str | None = Field(default=None, max_length=1000)
    title: str = "New Text2Model Forge asset"
    state: Literal[
        "created", "running", "awaiting_review", "blocked", "failed", "completed"
    ] = "created"
    current_stage: str = "D0"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    model: str = "qwen3_6_27b"
    comfy_url: str = "http://127.0.0.1:8188"
    localdeploy_url: str = "http://127.0.0.1:8000/v1"
    # "auto" prefers native Qwen Image 2512 text-to-image generation when its
    # model trio is present in ComfyUI, while keeping the existing SDXL route
    # usable on a machine that has not installed Qwen yet.
    concept_backend: Literal["auto", "qwen_image_2512", "qwen_image_edit_2511", "sdxl", "z_image_turbo"] = "auto"
    checkpoint: str = "dreamshaper_xl_v2_turbo.safetensors"
    style_lora: str | None = None
    style_lora_strength: float = Field(default=0.8, ge=0.0, le=1.5)
    style_lora_trigger: str | None = None
    prop_lora: str | None = None
    prop_lora_strength: float = Field(default=0.6, ge=0.0, le=1.5)
    # Defaults match concept_workflow()'s own long-standing steps/cfg defaults
    # in studio_comfy.py exactly, so a run created with no quality override
    # behaves identically to before these fields existed. See
    # src/text2model_forge/settings.py's quality_overrides() for how [asset].quality
    # picks a [quality.<tier>] section that can change these two.
    concept_steps: int = Field(default=30, ge=1, le=150)
    concept_cfg: float = Field(default=6.0, gt=0, le=30)
    # How D0 compiles the description into a spec, and how long any single
    # LLM call may take. "chunked" exists so a 7-8B local model can satisfy
    # StudioAssetSpec at all -- see src/text2model_forge/chunked_spec.py. Stored per run
    # so a resumed run keeps the strategy it was compiled under.
    spec_strategy: Literal["monolithic", "chunked"] = "monolithic"
    llm_timeout_seconds: float = Field(default=120, gt=0, le=3600)
    # Evict the other GPU service before each LLM / ComfyUI call. Off by
    # default because it costs a model reload every switch; needed on a card
    # too small to hold the reviewer and the image model at once. See
    # _VramHandoff in studio_pipeline.py.
    vram_handoff: bool = False
    # GPU compute only forbids explicit CPU inference while still allowing
    # orchestration, file I/O, and deterministic geometry operations on CPU.
    # strict_device_only additionally blocks any stage without a qualified GPU
    # implementation.
    device_policy: Literal["prefer_gpu", "gpu_compute_only", "strict_device_only"] = "prefer_gpu"
    gpu_safety_margin_gb: float = Field(default=0.75, ge=0, le=8)
    gpu_unload_timeout_seconds: float = Field(default=45, gt=0, le=300)
    automatic_retry_limit: int = Field(default=2, ge=0, le=5)
    # D1's quality budget expands time, not peak memory: every candidate is
    # generated sequentially and only the best bounded subset reaches the VLM.
    concept_candidates: int = Field(default=2, ge=2, le=12)
    concept_review_limit: int = Field(default=3, ge=1, le=6)
    concept_width: int = Field(default=768, ge=256, le=2048)
    concept_height: int = Field(default=1024, ge=256, le=2048)
    concept_min_quality_score: float = Field(default=0.35, ge=0, le=1)
    vae_tiling: bool = False
    # Which profiles/<name>.toml this run resolves its per-stage settings
    # from. Stored so a resumed run keeps the configuration it started with
    # rather than silently adopting whatever the profile says today.
    profile: str = Field(default="simple", pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    spec: StudioAssetSpec | None = None
    stages: list[StudioStageState]
    event_count: int = 0
    # A visibility flag only -- hides a run from the dashboard's default view.
    # Deliberately not a delete: this system's evidence and decisions are
    # append-only (see AGENTS.md's Human Gate Invariants), so there is no
    # "remove a run" action, only "stop showing it by default." Archiving
    # never changes `state`, is reversible, and StudioStore.list() still
    # returns archived runs -- callers that need every run regardless of
    # visibility (recover_interrupted_runs, artifact serving) must not filter
    # on this; only the dashboard and the CLI's `list` do, and only by choice.
    archived: bool = False

    def stage(self, stage_id: str) -> StudioStageState:
        for item in self.stages:
            if item.stage_id == stage_id:
                return item
        raise KeyError(stage_id)

    def compilation_brief(self) -> str:
        """Full D0 input without mutating the user's original description."""
        context: list[str] = []
        if self.intended_use:
            context.append(f"Intended use: {self.intended_use}")
        if self.must_have_features:
            context.append(f"Must-have features: {self.must_have_features}")
        if not context:
            return self.description
        return self.description + "\n\n" + "\n".join(context)


def new_studio_run(
    run_id: str, description: str, overrides: dict[str, object] | None = None
) -> StudioRun:
    """Create a run. `overrides` (see text2model_forge.settings.studio_overrides) may set
    any of StudioRun's own configuration fields; run_id/description/stages are
    always computed here and cannot be overridden this way."""
    return StudioRun(
        **(overrides or {}),
        run_id=run_id,
        source_revision=source_revision(),
        description=description.strip(),
        stages=[
            StudioStageState(stage_id=stage_id, label=label, gate_required=gate)
            for stage_id, label, gate in STAGE_DEFINITIONS
        ],
    )
