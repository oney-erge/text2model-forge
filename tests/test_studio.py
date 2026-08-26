from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading
import time

from PIL import Image
import pytest

from text2model_forge.studio_comfy import (
    concept_workflow,
    explicit_cpu_nodes,
    inpaint_workflow,
    make_chroma_alpha,
    make_footman_equipment_layout_guide,
    make_humanoid_openpose_guide,
    qwen_image_2512_workflow,
    qwen_image_edit_2511_workflow,
    z_image_turbo_workflow,
)
from text2model_forge.studio_models import (
    STAGE_DEFINITIONS,
    StudioAssetSpec,
    StudioCharacterSpec,
    StudioComponent,
    StudioEquipment,
    StudioQwenReview,
    StudioRun,
)
from text2model_forge.studio_pipeline import (
    StudioCoordinator,
    _VramHandoff,
    _composite_inpaint_crop,
    _prepare_inpaint_crop,
)
from text2model_forge.schemas import (
    Text2ModelLocalConfig,
    ExternalWorkerOutput,
    ExternalWorkerRequest,
    ExternalWorkerResponse,
    WorkerBinding,
)
from text2model_forge.studio_qwen import ConceptCorrectionPlan, ConceptPlan, StudioQwen
from text2model_forge.studio_qwen import RigidPartPlan, RigidStructurePlan, _history
from text2model_forge.studio_store import StudioConflictError, StudioStore


DESCRIPTION = (
    "An original chunky heroic dark-fantasy footman with one arming sword in his right hand "
    "and one broad shield on his left arm."
)


def footman_spec() -> StudioCharacterSpec:
    return StudioCharacterSpec(
        asset_id="original_footman",
        title="Original Iron Footman",
        description=DESCRIPTION,
        creative_direction="Original stylized broad mobile-readable defender.",
        anatomy_family="humanoid",
        height_m=1.82,
        silhouette=["broad shoulders", "large shield"],
        materials=["worn steel", "blue cloth", "leather"],
        equipment=[
            StudioEquipment(
                equipment_id="sword",
                category="weapon",
                side="right",
                socket="hand_right.grip",
                grip="palm_and_fingers",
                description="Original short arming sword.",
            ),
            StudioEquipment(
                equipment_id="shield",
                category="shield",
                side="left",
                socket="forearm_left.shield",
                grip="forearm_strap",
                description="Original broad strapped shield.",
            ),
        ],
        animations=["idle", "walk", "attack", "hit", "death", "block"],
        locked_features=["sword right", "shield left"],
        negative_constraints=["no copied IP", "no wrong handedness"],
        gameplay_readability=["equipment readable at sprite scale"],
    )


class FakeQwen:
    correction_calls = 0

    def compile_spec(self, description):
        assert description == DESCRIPTION
        return footman_spec()

    def concept_plan(self, spec, stage):
        return ConceptPlan(
            positive_prompt="A " * 45 + "complete original stylized footman, right sword, left shield.",
            negative_prompt="cropped, wrong hands, copied design",
            seeds=[101 + stage.iteration, 202 + stage.iteration],
            rationale="Compare two deterministic seeds.",
        )

    def review_concepts(self, spec, stage, images, *, comparison_board=None):
        ids = [item[0] for item in images]
        return StudioQwenReview(
            review_id=f"review-{stage.iteration}",
            stage_id="D1",
            iteration=stage.iteration,
            summary="Candidate one has the clearer right sword and left shield.",
            strengths=["readable equipment"],
            issues=["check grip close-up"],
            candidate_ranking=ids,
            recommended_evidence_id=ids[0],
            confidence=0.8,
        )

    def concept_correction_plan(self, spec, stage, candidate_ids, comparison_board=None):
        self.correction_calls += 1
        assert stage.human_decisions[-1].comment == "Make the shield larger."
        return ConceptCorrectionPlan(
            operation_id="regenerate_complete_asset",
            base_evidence_id=candidate_ids[0],
            edit_box_normalized=[0.0, 0.0, 1.0, 1.0],
            positive_prompt="A " * 45 + "complete original footman with a larger left shield and right sword.",
            negative_prompt="small shield, missing shield, missing sword, wrong hands",
            seeds=[303, 404],
            denoise=0.8,
            diagnosis="Shield was too small.",
            preserve=["Right-hand sword."],
        )

    def review_geometry(self, spec, stage, selected_concept, diagnostic, metrics):
        review = StudioQwenReview(
            review_id=f"d2-review-{stage.iteration}",
            stage_id="D2",
            iteration=stage.iteration,
            summary="Geometry candidate preserves the approved silhouette and passes numeric gates.",
            strengths=["watertight", "proportions match the approved concept"],
            issues=[],
            candidate_ranking=["geometry-candidate"],
            recommended_evidence_id="geometry-candidate",
            confidence=0.85,
            hard_requirements_satisfied=True,
        )
        return review, True

    def review_cleanup(self, spec, stage, selected_concept, diagnostic, metrics):
        review = StudioQwenReview(
            review_id=f"d3-review-{stage.iteration}",
            stage_id="D3",
            iteration=stage.iteration,
            summary="Cleanup preserved identity; no floating or missing structural pieces.",
            strengths=["single connected component", "silhouette unchanged"],
            issues=[],
            candidate_ranking=["cleanup-candidate"],
            recommended_evidence_id="cleanup-candidate",
            confidence=0.85,
            hard_requirements_satisfied=True,
        )
        return review, True

    def review_deformable_rig(self, spec, stage, selected_concept, stress_board, metrics):
        return StudioQwenReview(
            review_id=f"d4-review-{stage.iteration}",
            stage_id="D4",
            iteration=stage.iteration,
            summary="Rig landmarks and stress poses are credible for the approved identity.",
            strengths=["no collapsed shoulder", "bounded influences"],
            issues=[],
            candidate_ranking=["rig-stress-board"],
            recommended_evidence_id="rig-stress-board",
            confidence=0.8,
            hard_requirements_satisfied=True,
        )


class FakeComfy:
    def __init__(self) -> None:
        self.workflows: list[dict] = []

    def checkpoints(self):
        return ["dreamshaper_xl_v2_turbo.safetensors"]

    def controlnets(self):
        return ["controlnet_openpose_sdxl_xinsir.safetensors"]

    def models(self, kind: str):
        return ["Warcraft style.safetensors"] if kind == "loras" else []

    def upload_image(self, name: str, data: bytes, subfolder: str = "text2model_studio") -> str:
        return f"{subfolder}/{name}"

    def generate(self, *, workflow, destination: Path, timeout_seconds=900):
        self.workflows.append(workflow)
        destination.mkdir(parents=True, exist_ok=True)
        # A workflow ending in SaveGLB produces a mesh, not an image -- mirror
        # real ComfyUI, where Hunyuan3D writes a .glb rather than a .png.
        if any(node["class_type"] == "SaveGLB" for node in workflow.values()):
            return [_synthetic_mesh(destination / "hunyuan3d_mesh.glb")]
        target = destination / "concept.png"
        # An edge-connected green border around a non-green center, so this
        # image is also valid input to make_chroma_alpha() (D2's geometry
        # seed path key it out) -- not just any RGB PNG, exactly like
        # test_chroma_alpha_removes_only_edge_connected_green's fixture.
        image = Image.new("RGB", (64, 96), (0, 220, 0))
        for y in range(16, 80):
            for x in range(12, 52):
                image.putpixel((x, y), (70, 90, 120))
        image.save(target)
        return [target]


def _synthetic_mesh(path: Path) -> Path:
    """A genuinely valid GLB, not a placeholder file. The stages that consume
    these actually parse them with trimesh and compute real geometric
    properties (vertices, faces, watertightness), so an empty file would fail
    for the wrong reason. subdivisions=4 gives 2562 vertices / 5120 faces,
    clearing D2's >=1000/>=1000 hard gate by a wide margin, and is cheap."""
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=4).export(path)
    return path


def _synthetic_image(path: Path, color: tuple[int, int, int] = (90, 100, 80)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return path


def _synthetic_report(path: Path, **values) -> Path:
    """A report JSON whose gate field _adopt_d4_output actually reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gate_passed": True, **values}), encoding="utf-8")
    return path


class FakeWorkerExecutor:
    """Stands in for _execute_worker's real path (config.local.toml lookup +
    a real subprocess via WorkerManager/SubprocessWorkerAdapter), the same way
    FakeQwen/FakeComfy stand in for their services.

    Each operation_id gets a fixture producing the exact output roles the
    corresponding stage consumes -- these are contracts, not arbitrary
    placeholders: D3 requires a role="candidate_geometry" GLB, and
    _adopt_d4_output requires specific report roles at D5/D6, so getting
    them wrong fails the stage exactly as a real misbehaving worker would."""

    def __init__(self) -> None:
        self.requests: list[ExternalWorkerRequest] = []

    def __call__(self, worker_id: str, request: ExternalWorkerRequest, *, timeout_seconds: float):
        self.requests.append(request)
        output_root = Path(request.output_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        handler = getattr(self, f"_op_{request.operation_id.replace('.', '_')}", None)
        if handler is None:
            raise AssertionError(
                f"FakeWorkerExecutor has no fixture for worker_id={worker_id!r} "
                f"operation_id={request.operation_id!r}"
            )
        outputs, diagnostics = handler(request, output_root)
        return ExternalWorkerResponse(
            job_id=request.job_id,
            status="succeeded",
            outputs=outputs,
            diagnostics={"synthetic": True, **diagnostics},
        )

    def _op_geometry_generate_from_rgba(self, request, output_root):
        glb = _synthetic_mesh(output_root / "geometry_candidate.glb")
        return (
            [ExternalWorkerOutput(path=str(glb), media_type="model/gltf-binary", role="geometry_candidate")],
            {},
        )

    def _op_blender_repair(self, request, output_root):
        front = _synthetic_image(output_root / "candidate_front.png")
        geometry = _synthetic_mesh(output_root / "candidate_geometry.glb")
        checkpoint = output_root / "candidate_checkpoint.blend"
        checkpoint.write_bytes(b"synthetic-blend-checkpoint")
        return (
            [
                ExternalWorkerOutput(path=str(front), media_type="image/png", role="candidate_front"),
                ExternalWorkerOutput(
                    path=str(geometry), media_type="model/gltf-binary", role="candidate_geometry"
                ),
                ExternalWorkerOutput(
                    path=str(checkpoint),
                    media_type="application/x-blender",
                    role="candidate_checkpoint",
                ),
            ],
            {"hard_gate_passed": True},
        )

    def _op_blender_propose_short_biped_rig(self, request, output_root):
        """D4's rig probe. Must emit every role D4's stress board needs
        (image/* with a rig_ prefix) plus the report/checkpoint roles that
        _adopt_d4_output later requires at D5 and D6 -- those two stages
        run no worker of their own and adopt these directly."""
        outputs = [
            ExternalWorkerOutput(
                path=str(_synthetic_image(output_root / "rig_neutral.png")),
                media_type="image/png",
                role="rig_neutral",
            ),
            ExternalWorkerOutput(
                path=str(_synthetic_image(output_root / "rig_shoulder_stress.png")),
                media_type="image/png",
                role="rig_shoulder_stress",
            ),
            ExternalWorkerOutput(
                path=str(_synthetic_mesh(output_root / "rigged_candidate.glb")),
                media_type="model/gltf-binary",
                role="rigged_candidate",
            ),
        ]
        rigged_checkpoint = output_root / "rigged_candidate_checkpoint.blend"
        rigged_checkpoint.write_bytes(b"synthetic-rigged-blend")
        outputs.append(
            ExternalWorkerOutput(
                path=str(rigged_checkpoint),
                media_type="application/x-blender",
                role="rigged_candidate_checkpoint",
            )
        )
        for role in (
            "rig_contract",
            "landmarks_contract",
            "skinning_report",
            "deformation_report",
            "neutral_comparison_report",
            "rigged_export_validation",
        ):
            outputs.append(
                ExternalWorkerOutput(
                    path=str(_synthetic_report(output_root / f"{role}.json", role=role)),
                    media_type="application/json",
                    role=role,
                )
            )
        return outputs, {"maximum_influences": 4, "hard_gate_passed": True}

    def _op_blender_render_diagnostics(self, request, output_root):
        """D9's static/prop delivery render. Every image output becomes a cell
        in the delivery board, and the board being non-empty IS D9's automatic
        gate -- returning zero images fails the stage."""
        return (
            [
                ExternalWorkerOutput(
                    path=str(_synthetic_image(output_root / f"delivery_{view}.png")),
                    media_type="image/png",
                    role=f"delivery_{view}",
                )
                for view in ("front", "side", "back", "top")
            ],
            {"hard_gate_passed": True},
        )


class FakeScriptRunner:
    """Stands in for the adapters/ helper scripts invoked via subprocess.run
    (D8's surface bake and review, D7/D9-deformable's motion chain).

    Deliberately delegates render_glb_diagnostic.py to the REAL subprocess:
    that script needs only trimesh/numpy/PIL, so a test can and should run it
    for real rather than pretend. Only the Blender/ComfyUI-dependent scripts
    are simulated."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command, *, text=True, capture_output=True, timeout=None):
        self.commands.append(list(command))
        script = next((part for part in command if str(part).endswith(".py")), "")
        name = Path(str(script)).name
        if name == "render_glb_diagnostic.py":
            return subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        handler = getattr(self, f"_script_{name[:-3]}", None)
        if handler is None:
            raise AssertionError(f"FakeScriptRunner has no fixture for script {name!r}")
        handler(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    @staticmethod
    def _argument(command, flag: str) -> str:
        return str(command[command.index(flag) + 1])

    def _script_bake_surface(self, command):
        surface = Path(self._argument(command, "--output-directory"))
        surface.mkdir(parents=True, exist_ok=True)
        _synthetic_image(surface / "surface_review.png")
        (surface / "text2model_surface_master.blend").write_bytes(b"synthetic-surface-master")
        (surface / "surface_validation.json").write_text(
            json.dumps({"automatic_gate_passed": True, "image_metrics": {"visible_luminance": 0.42}}),
            encoding="utf-8",
        )

    def _script_review_surface_master(self, command):
        surface = Path(self._argument(command, "--surface-directory"))
        surface.mkdir(parents=True, exist_ok=True)
        (surface / "qwen_surface_mediator.json").write_text(
            json.dumps({"corrected_overall": "ready_for_final_render"}), encoding="utf-8"
        )

    def _script_run_motion_candidate_pipeline(self, command):
        """The resumable D7->D10 chain. --stop-after names how far it should
        go, and each stage checks for the exact evidence its own step
        publishes, so this writes progressively more as the stop point
        advances -- mirroring the real chain's resumable contract."""
        root = Path(self._argument(command, "--output-root"))
        stop_after = self._argument(command, "--stop-after")
        retarget = root / "retarget"
        review = retarget / "human_review"
        _synthetic_image(review / "all_motion_front_keyposes.png")
        for name in ("attack", "walk", "death"):
            _synthetic_image(review / f"{name}_front_keyposes.png")
        (retarget / "quaternius_retargeted_candidate.blend").write_bytes(b"synthetic-retargeted")
        (retarget / "retarget_validation.json").write_text(
            json.dumps({"automatic_gate_passed": True}), encoding="utf-8"
        )
        (review / "qwen_retarget_mediator.json").write_text(
            json.dumps({"corrected_overall": "ready_for_human_gate", "reason": "Motion is clean."}),
            encoding="utf-8",
        )
        if stop_after == "retarget_qwen_review":
            return
        package = root / "sprites" / "package"
        _synthetic_image(package / "sprite_review.png")
        (package / "candidate_unit_manifest.json").write_text(
            json.dumps({"automatic_gate_passed": True, "source_master_sha256": "0" * 64}),
            encoding="utf-8",
        )
        (package / "qwen_sprite_mediator.json").write_text(
            json.dumps({"corrected_overall": "ready_for_unity_candidate"}), encoding="utf-8"
        )
        if stop_after == "sprite_qwen_review":
            return
        bundle = root / "unity_smoke_bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "review.html").write_text("<h1>synthetic bundle</h1>", encoding="utf-8")
        (bundle / "bundle_manifest.json").write_text(
            json.dumps({"human_approved": False}), encoding="utf-8"
        )


class QwenImageFakeComfy(FakeComfy):
    def __init__(self) -> None:
        self.workflows: list[dict] = []

    def models(self, kind: str):
        values = {
            "diffusion_models": ["qwen_image_2512_fp8_e4m3fn.safetensors"],
            "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
            "vae": ["qwen_image_vae.safetensors"],
        }
        return values.get(kind, [])

    # generate() is inherited from FakeComfy, which already records into
    # self.workflows -- do not also append here, or every call double-counts.


class ZImageTurboFakeComfy(FakeComfy):
    def __init__(self) -> None:
        self.workflows: list[dict] = []

    def models(self, kind: str):
        values = {
            "diffusion_models": ["z_image_turbo_int8_convrot.safetensors"],
            "text_encoders": ["qwen_3_4b_fp8_mixed.safetensors"],
            "vae": ["ae.safetensors"],
        }
        return values.get(kind, [])


class BothGeneratorsFakeComfy(FakeComfy):
    """A machine with Qwen Image 2512 *and* Z-Image Turbo installed, which is
    the only situation where `auto`'s preference between them is observable."""

    def __init__(self) -> None:
        self.workflows: list[dict] = []

    def models(self, kind: str):
        values = {
            "diffusion_models": [
                "qwen_image_2512_fp8_e4m3fn.safetensors",
                "z_image_turbo_int8_convrot.safetensors",
            ],
            "text_encoders": [
                "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "qwen_3_4b_fp8_mixed.safetensors",
            ],
            "vae": ["qwen_image_vae.safetensors", "ae.safetensors"],
        }
        return values.get(kind, [])


class ControlFakeComfy(QwenImageFakeComfy):
    def __init__(self) -> None:
        super().__init__()
        self.interrupt_calls = 0
        self.memory_releases: list[tuple[bool, bool]] = []

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        self.memory_releases.append((unload_models, free_memory))


class BlockingControlFakeComfy(ControlFakeComfy):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.interrupted = threading.Event()

    def generate(self, *, workflow, destination: Path, timeout_seconds=900):
        self.workflows.append(workflow)
        self.started.set()
        if not self.interrupted.wait(timeout=3):
            raise AssertionError("test did not request a ComfyUI interrupt")
        raise RuntimeError("ComfyUI workflow interrupted")

    def interrupt(self) -> None:
        super().interrupt()
        self.interrupted.set()


def wait_for(store: StudioStore, run_id: str, state: str, timeout: float = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = store.load(run_id)
        if run.state == state:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run never reached {state}: {store.load(run_id).model_dump()}")


_SETTLED_STAGE_STATES = {
    "approved",
    "rejected",
    "failed",
    "skipped",
    "blocked",
    # A gated stage that finished its work and is waiting on a human is
    # settled too -- for D10 that is the normal, successful end of a run.
    "awaiting_review",
}


def _wait_for_stage_settled(store: StudioStore, run_id: str, stage_id: str, timeout: float = 5):
    """Wait for one stage to stop doing work. Unlike wait_for(), which watches
    run.state, this watches a specific stage -- needed for stages like D2 that
    have no human gate and so never park the run in awaiting_review."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = store.load(run_id)
        if run.stage(stage_id).state in _SETTLED_STAGE_STATES:
            return run
        time.sleep(0.02)
    raise AssertionError(
        f"{stage_id} never settled: {store.load(run_id).stage(stage_id).model_dump()}"
    )


def test_stage_contract_covers_d0_through_d10() -> None:
    assert [item[0] for item in STAGE_DEFINITIONS] == [f"D{i}" for i in range(11)]
    assert [item[0] for item in STAGE_DEFINITIONS if item[2]] == ["D1", "D4", "D7", "D8", "D10"]


def test_workflow_uses_only_qualified_core_nodes() -> None:
    workflow = concept_workflow(
        checkpoint="model.safetensors",
        positive="original footman",
        negative="copied design",
        seed=42,
        prefix="Text2ModelStudio/test",
    )
    assert {item["class_type"] for item in workflow.values()} == {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }
    assert workflow["4"]["inputs"] == {"width": 768, "height": 1024, "batch_size": 1}


def test_inpaint_workflow_uses_bounded_core_mask() -> None:
    workflow = inpaint_workflow(
        checkpoint="model.safetensors",
        positive="A " * 45,
        negative="missing equipment and wrong grip",
        seed=42,
        prefix="Text2ModelStudio/test",
        source_image="studio/source.png",
        mask_image="studio/mask.png",
        denoise=0.7,
    )
    assert workflow["5"] == {
        "class_type": "LoadImageMask",
        "inputs": {"image": "studio/mask.png", "channel": "red"},
    }
    assert workflow["6"]["class_type"] == "VAEEncodeForInpaint"
    assert workflow["6"]["inputs"]["grow_mask_by"] == 12
    assert workflow["7"]["class_type"] == "DifferentialDiffusion"
    assert workflow["8"]["inputs"]["denoise"] == 0.7


def test_hunyuan3d_workflow_matches_comfyuis_own_image_to_model_template() -> None:
    """Every assertion here is a bug found only by running this against real
    ComfyUI, and each produced garbage rather than an error:

    - resolution must be the DiT latent resolution (3072), not the VAE's
      octree resolution (256). Passing 256 yielded a mesh of 53,560
      disconnected fragments.
    - the mesher must be VoxelToMesh "surface net". VoxelToMeshBasic emits
      per-voxel triangles that arrive as a shredded soup.
    - Hunyuan3Dv2Conditioning takes a CLIP_VISION_OUTPUT, so the image has to
      pass through CLIPVisionEncode first. Wiring the checkpoint's CLIP_VISION
      slot straight in is a type mismatch ComfyUI rejects with HTTP 400.
    """
    from text2model_forge.studio_comfy import hunyuan3d_workflow

    workflow = hunyuan3d_workflow(
        image="subject.png", prefix="test/mesh", seed=7, checkpoint="hunyuan3d-dit-v2_fp16.safetensors"
    )
    by_class = {node["class_type"]: node for node in workflow.values()}

    assert by_class["EmptyLatentHunyuan3Dv2"]["inputs"]["resolution"] == 3072
    assert by_class["VoxelToMesh"]["inputs"]["algorithm"] == "surface net"
    assert "VoxelToMeshBasic" not in by_class
    assert by_class["CLIPVisionEncode"]["inputs"]["crop"] == "none"

    # The conditioning must consume the ENCODER's output, not the raw model.
    encode_id = next(k for k, n in workflow.items() if n["class_type"] == "CLIPVisionEncode")
    assert by_class["Hunyuan3Dv2Conditioning"]["inputs"]["clip_vision_output"] == [encode_id, 0]
    # ...and that encoder must read the uploaded image, not an empty latent.
    load_id = next(k for k, n in workflow.items() if n["class_type"] == "LoadImage")
    assert by_class["CLIPVisionEncode"]["inputs"]["image"] == [load_id, 0]
    assert by_class["LoadImage"]["inputs"]["image"] == "subject.png"


def test_chroma_alpha_flattens_the_backdrop_to_white_in_the_colour_channels(tmp_path: Path) -> None:
    """Image-to-3D reads RGB through CLIP-Vision and ignores alpha. Leaving
    the keyed-out backdrop in the colour channels at alpha=0 made Hunyuan3D
    reconstruct it as a giant flat slab wrapped around the subject, so the
    backdrop must be flattened to white in RGB as well as cleared in alpha."""
    source = tmp_path / "subject.png"
    image = Image.new("RGB", (80, 80), (0, 210, 0))
    for y in range(20, 60):
        for x in range(20, 60):
            image.putpixel((x, y), (120, 110, 105))
    image.save(source)

    output = tmp_path / "keyed.png"
    make_chroma_alpha(source, output)
    result = Image.open(output)
    assert result.mode == "RGBA"
    rgb = result.convert("RGB")
    assert rgb.getpixel((2, 2)) == (255, 255, 255), "backdrop must be white in RGB, not left green"
    assert result.getpixel((2, 2))[3] == 0, "backdrop must still be transparent in alpha"
    assert rgb.getpixel((40, 40)) == (120, 110, 105), "the subject must be untouched"
    assert result.getpixel((40, 40))[3] == 255


def test_concept_workflow_passes_steps_and_cfg_through_to_the_sampler() -> None:
    workflow = concept_workflow(
        checkpoint="model.safetensors",
        positive="original footman",
        negative="copied design",
        seed=42,
        prefix="Text2ModelStudio/test",
        steps=45,
        cfg=7.0,
    )
    samplers = [item for item in workflow.values() if item["class_type"] == "KSampler"]
    assert samplers[0]["inputs"]["steps"] == 45
    assert samplers[0]["inputs"]["cfg"] == 7.0


def test_qwen_image_edit_workflow_uses_native_dual_image_conditioning() -> None:
    workflow = qwen_image_edit_2511_workflow(
        prompt="Preserve the footman and fix only the sword grip.",
        negative_prompt="extra character, equipment rack",
        seed=42,
        prefix="Text2ModelStudio/test",
        source_image="studio/source.png",
    )
    assert workflow["1"]["class_type"] == "UNETLoader"
    assert workflow["4"]["inputs"]["type"] == "qwen_image"
    assert workflow["4"]["inputs"]["device"] == "default"
    assert workflow["8"]["class_type"] == "TextEncodeQwenImageEdit"
    assert workflow["8"]["inputs"]["image"] == ["7", 0]
    assert workflow["12"]["class_type"] == "VAEEncode"
    assert explicit_cpu_nodes(workflow) == []
    assert workflow["13"]["inputs"]["steps"] == 40


def test_qwen_image_2512_workflow_is_native_portrait_text_to_image() -> None:
    workflow = qwen_image_2512_workflow(
        prompt="One original human footman with a sword and shield.",
        negative_prompt="pixel art, duplicate character",
        seed=42,
        prefix="Text2ModelStudio/test",
    )
    assert workflow["1"]["inputs"]["unet_name"] == "qwen_image_2512_fp8_e4m3fn.safetensors"
    assert workflow["4"]["inputs"] == {
        "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "type": "qwen_image",
        "device": "default",
    }
    assert workflow["8"]["inputs"] == {"width": 1104, "height": 1472, "batch_size": 1}
    assert workflow["9"]["inputs"]["steps"] == 50


def test_z_image_turbo_workflow_is_native_portrait_text_to_image() -> None:
    workflow = z_image_turbo_workflow(
        prompt="One original human footman with a sword and shield.",
        negative_prompt="pixel art, duplicate character",
        seed=42,
        prefix="Text2ModelStudio/test",
    )
    assert workflow["1"]["class_type"] == "UNETLoader"
    assert workflow["1"]["inputs"]["unet_name"] == "z_image_turbo_int8_convrot.safetensors"
    assert workflow["2"]["inputs"] == {
        "clip_name": "qwen_3_4b_fp8_mixed.safetensors",
        "type": "stable_diffusion",
        "device": "default",
    }
    assert workflow["3"]["inputs"]["vae_name"] == "ae.safetensors"
    # turbo/distilled settings, not the non-distilled defaults: few steps, cfg=1
    assert workflow["7"]["inputs"]["steps"] == 8
    assert workflow["7"]["inputs"]["cfg"] == 1.0
    assert workflow["6"]["inputs"] == {"width": 1104, "height": 1472, "batch_size": 1}


def test_tiled_vae_decode_is_available_without_changing_the_sampler() -> None:
    workflow = concept_workflow(
        checkpoint="model.safetensors",
        positive="one isolated asset",
        negative="clutter",
        seed=42,
        prefix="Text2ModelStudio/test",
        tiled_vae=True,
    )
    assert workflow["6"]["class_type"] == "VAEDecodeTiled"
    assert workflow["6"]["inputs"]["tile_size"] == 512
    assert workflow["5"]["class_type"] == "KSampler"


def test_concept_workflow_chains_loras_and_pose_control_around_one_body() -> None:
    workflow = concept_workflow(
        checkpoint="model.safetensors",
        positive="complete original footman, right hand sword, left arm shield",
        negative="missing equipment, duplicate body",
        seed=42,
        prefix="Text2ModelStudio/test",
        loras=[("style.safetensors", 0.85), ("armor.safetensors", 0.6)],
        control_guides=[("openpose.safetensors", "pose.png", 0.85, 0.0, 0.85)],
    )
    lora_nodes = [item for item in workflow.values() if item["class_type"] == "LoraLoader"]
    controls = [item for item in workflow.values() if item["class_type"] == "ControlNetApplyAdvanced"]
    samplers = [item for item in workflow.values() if item["class_type"] == "KSampler"]
    assert [item["inputs"]["lora_name"] for item in lora_nodes] == ["style.safetensors", "armor.safetensors"]
    assert [item["inputs"]["strength_model"] for item in lora_nodes] == [0.85, 0.6]
    assert [item["inputs"]["strength"] for item in controls] == [0.85]
    last_lora_id = next(
        node_id for node_id, item in workflow.items() if item is lora_nodes[-1]
    )
    # The LoRA chain must feed the sampler's model input -- there is only ever one figure, never
    # a second region-conditioned body.
    assert samplers[0]["inputs"]["model"] == [last_lora_id, 0]
    assert "ConditioningSetArea" not in {item["class_type"] for item in workflow.values()}
    assert "ConditioningCombine" not in {item["class_type"] for item in workflow.values()}


def test_concept_workflow_with_no_lora_or_control_matches_prior_core_node_shape() -> None:
    workflow = concept_workflow(
        checkpoint="model.safetensors",
        positive="original footman",
        negative="copied design",
        seed=42,
        prefix="Text2ModelStudio/test",
    )
    assert {item["class_type"] for item in workflow.values()} == {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }


def test_humanoid_guide_and_local_crop_preserve_unmasked_pixels(tmp_path: Path) -> None:
    pose = make_humanoid_openpose_guide(tmp_path / "guides")
    assert Image.open(pose).size == (768, 1024)

    source = tmp_path / "source.png"
    Image.new("RGB", (300, 400), "blue").save(source)
    crop, mask, crop_box = _prepare_inpaint_crop(
        source,
        [0.05, 0.2, 0.45, 0.8],
        tmp_path / "repair",
    )
    assert Image.open(crop).size == (768, 1024)
    generated = tmp_path / "generated.png"
    Image.new("RGB", (768, 1024), "red").save(generated)
    output = _composite_inpaint_crop(source, generated, mask, crop_box, tmp_path / "output.png")
    with Image.open(output).convert("RGB") as result:
        assert result.getpixel((299, 399)) == (0, 0, 255)
        assert result.getpixel((75, 200))[0] > result.getpixel((75, 200))[2]


def test_footman_equipment_layout_guide_has_a_fixed_portrait_canvas(tmp_path: Path) -> None:
    guide = make_footman_equipment_layout_guide(tmp_path / "footman_layout.png")
    assert Image.open(guide).size == (768, 1024)


def test_generic_static_asset_contract_accepts_wall_without_character_fields() -> None:
    wall = StudioAssetSpec(
        asset_id="fortress_wall",
        title="Original Fortress Wall",
        description="A worn modular defensive wall.",
        creative_direction="Readable original stylized stone construction.",
        asset_kind="architecture",
        behavior="static",
        anatomy_family=None,
        height_m=3.0,
        dimensions_m=[4.0, 3.0, 0.8],
        silhouette=["crenellated top"],
        materials=["worn stone"],
        animations=[],
        locked_features=["modular ends"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["doorway scale readable"],
    )
    assert wall.asset_kind == "architecture"
    assert wall.behavior == "static"


class _SpecQwen:
    """Minimal fake Qwen that returns a caller-supplied spec, for testing D0's
    asset_kind/behavior-driven stage-skip logic without needing D1-D10 to
    actually work for a non-character asset."""

    def __init__(self, spec: StudioAssetSpec) -> None:
        self._spec = spec

    def compile_spec(self, description):
        return self._spec


def _run_d0_only(spec: StudioAssetSpec, run_id: str, tmp_path: Path) -> StudioRun:
    store = StudioStore(tmp_path)
    run = store.create(run_id, spec.description)
    coordinator = StudioCoordinator(store, qwen_factory=lambda run: _SpecQwen(spec))
    coordinator._run_d0(run)
    store.save(run)
    return store.load(run_id)


def test_static_prop_skips_only_rig_and_motion_stages(tmp_path: Path) -> None:
    chair = StudioAssetSpec(
        asset_id="oak_chair",
        title="Original Oak Chair",
        description="A simple original sturdy wooden dining chair with a straight back.",
        creative_direction="Readable original cottage-style furniture, no franchise motifs.",
        asset_kind="prop",
        behavior="static",
        anatomy_family=None,
        height_m=0.9,
        dimensions_m=[0.45, 0.9, 0.45],
        silhouette=["straight back", "four legs"],
        materials=["oak wood"],
        animations=[],
        locked_features=["four legs", "straight back"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["silhouette reads at prop scale"],
    )
    run = _run_d0_only(chair, "chair-v1", tmp_path)
    assert run.stage("D0").state == "approved"
    # a static prop still needs geometry, cleanup, surface, sprites, and validation
    for stage_id in ("D2", "D3", "D8", "D9", "D10"):
        assert run.stage(stage_id).state != "skipped", stage_id
        assert run.stage(stage_id).applicable is True, stage_id
    # but no skeleton, rig, deforming weights, or motion
    for stage_id in ("D4", "D5", "D6", "D7"):
        stage = run.stage(stage_id)
        assert stage.state == "skipped", stage_id
        assert stage.applicable is False, stage_id
        assert stage.progress == 1


def test_material_asset_skips_every_geometry_and_articulation_stage(tmp_path: Path) -> None:
    rust_iron = StudioAssetSpec(
        asset_id="rust_iron_material",
        title="Weathered Rust Iron",
        description="An original weathered rust-streaked iron surface material.",
        creative_direction="Readable original grunge metal, tileable at gameplay scale.",
        asset_kind="material",
        behavior="static",
        anatomy_family=None,
        height_m=None,
        silhouette=["flat tileable swatch"],
        materials=["rust", "iron"],
        animations=[],
        locked_features=["rust streak pattern"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["reads at tile scale"],
    )
    run = _run_d0_only(rust_iron, "material-v1", tmp_path)
    assert run.stage("D0").state == "approved"
    for stage_id in ("D2", "D3", "D4", "D5", "D6", "D7"):
        stage = run.stage(stage_id)
        assert stage.state == "skipped", stage_id
        assert stage.applicable is False, stage_id
    # a material still goes through concept, surface, sprites, and validation
    for stage_id in ("D1", "D8", "D9", "D10"):
        assert run.stage(stage_id).state != "skipped", stage_id


def test_rigid_articulated_asset_skips_skinning_and_motion_without_clips(tmp_path: Path) -> None:
    gate = StudioAssetSpec(
        asset_id="iron_gate",
        title="Original Hinged Iron Gate",
        description="An original hinged iron gate with a single swinging door.",
        creative_direction="Readable original ironwork, open/close states.",
        asset_kind="architecture",
        behavior="rigid_articulated",
        anatomy_family=None,
        height_m=2.4,
        dimensions_m=[1.6, 2.4, 0.2],
        silhouette=["vertical iron bars"],
        materials=["iron"],
        components=[
            StudioComponent(
                component_id="door",
                role="movable_part",
                connection="hinge to frame",
                motion="rigid",
                description="Single swinging door leaf.",
            )
        ],
        animations=[],
        locked_features=["single door leaf"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["open/close states readable"],
    )
    run = _run_d0_only(gate, "gate-v1", tmp_path)
    assert run.stage("D6").state == "skipped"
    assert run.stage("D6").message == "Rigid articulated assets do not require deforming skin weights."
    assert run.stage("D7").state == "skipped"
    assert run.stage("D7").message == "No rigid motion clips were requested."
    # rigid articulation still needs a skeleton/rig for its hinge, unlike a static prop
    assert run.stage("D4").state != "skipped"
    assert run.stage("D5").state != "skipped"


def test_rigid_structure_contract_normalizes_bounds_and_limits() -> None:
    plan = RigidStructurePlan(
        parts=[
            RigidPartPlan(
                component_id="door_left",
                front_box_normalized=[0.1, 0.2, 0.45, 0.9],
                pivot_normalized=[0.1, 0.55],
                rotation_axis="z",
                minimum_degrees=95,
                maximum_degrees=0,
                neutral_degrees=0,
                rationale="Side hinge.",
            )
        ],
        static_component_ids=["frame"],
        confidence=0.8,
    )
    assert plan.parts[0].minimum_degrees == 0
    assert plan.parts[0].maximum_degrees == 95


def test_stage_selector_treats_typed_skips_as_completed() -> None:
    from text2model_forge.studio_models import new_studio_run

    state = new_studio_run("static-wall", "An original static stone wall for a mobile game.")
    state.stage("D0").state = "approved"
    state.stage("D1").state = "approved"
    state.stage("D2").state = "approved"
    state.stage("D3").state = "approved"
    for stage_id in ("D4", "D5", "D6", "D7"):
        state.stage(stage_id).state = "skipped"
        state.stage(stage_id).applicable = False
    assert StudioCoordinator._next_stage(state).stage_id == "D8"


def test_numeric_history_stays_bounded_after_many_iterations() -> None:
    from text2model_forge.studio_models import new_studio_run

    stage = new_studio_run("history", DESCRIPTION).stage("D1")
    stage.iteration = 12
    for iteration in range(1, 13):
        stage.qwen_reviews.append(
            StudioQwenReview(
                review_id=f"review-{iteration}",
                stage_id="D1",
                iteration=iteration,
                summary="long diagnosis " * 200,
                issues=["issue " * 100],
                candidate_ranking=[f"candidate-{iteration}"],
                recommended_evidence_id=f"candidate-{iteration}",
                confidence=0.5,
            )
        )
    assert len(_history(stage)) < 12_000


def test_chroma_alpha_removes_only_edge_connected_green(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "rgba.png"
    image = Image.new("RGB", (100, 100), (0, 220, 0))
    for y in range(20, 80):
        for x in range(30, 70):
            image.putpixel((x, y), (80, 90, 120))
    # A green emblem inside the foreground must remain because it is not edge-connected.
    for y in range(40, 50):
        for x in range(45, 55):
            image.putpixel((x, y), (0, 200, 0))
    image.save(source)
    metrics = make_chroma_alpha(source, output)
    assert metrics["meaningful_alpha"] is True
    with Image.open(output).convert("RGBA") as result:
        assert result.getpixel((0, 0))[3] == 0
        assert result.getpixel((50, 45))[3] == 255


def test_explicit_handedness_validator_fails_closed() -> None:
    wrong = footman_spec().model_copy(deep=True)
    wrong.equipment[0].side = "left"
    with pytest.raises(ValueError, match="right-hand sword"):
        StudioQwen._validate_explicit_handedness(DESCRIPTION, wrong)


def test_pipeline_stops_for_review_and_rejection_reuses_comment(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    store.create("footman-v1", DESCRIPTION)
    qwen = FakeQwen()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: FakeComfy(),
            executor=executor,
        )
        assert coordinator.submit("footman-v1") is True
        first = wait_for(store, "footman-v1", "awaiting_review")
        assert first.current_stage == "D1"
        assert first.stage("D0").state == "approved"
        assert first.stage("D1").iteration == 1
        candidates = [
            item
            for item in first.stage("D1").evidence
            if item.media_type == "image/png" and item.metrics.get("selectable") is True
        ]
        assert len(candidates) == 2

        store.decide("footman-v1", "D1", "reject", "Make the shield larger.", candidates[0].evidence_id)
        coordinator.submit("footman-v1")
        second = wait_for(store, "footman-v1", "awaiting_review")
        assert second.stage("D1").iteration == 2
        assert qwen.correction_calls == 1
        assert len(second.stage("D1").human_decisions) == 1
        assert len(
            [
                item
                for item in second.stage("D1").evidence
                if item.media_type == "image/png" and item.metrics.get("selectable") is True
            ]
        ) == 4
        event_types = [item["event_type"] for item in store.read_events("footman-v1")]
        assert "gate_rejected" in event_types


def test_pipeline_prefers_installed_qwen_image_generation_for_a_typed_humanoid_brief(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("footman-qwen-v1", DESCRIPTION)
    run.concept_backend = "qwen_image_2512"
    store.save(run)
    qwen = FakeQwen()
    comfy = QwenImageFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit("footman-qwen-v1") is True
        result = wait_for(store, "footman-qwen-v1", "awaiting_review")
    stage = result.stage("D1")
    assert stage.metrics["workflow_strategy"] == "qwen_image_2512_t2i_v1"
    assert not any(item.evidence_id.endswith("qwen-edit-layout-guide") for item in stage.evidence)
    assert len(comfy.workflows) == 2
    assert all(workflow["1"]["class_type"] == "UNETLoader" for workflow in comfy.workflows)


def test_pipeline_uses_z_image_turbo_when_explicitly_selected(tmp_path: Path) -> None:
    """z_image_turbo is a deliberate, explicit-only choice -- unlike Qwen it is
    never picked by "auto" -- so this exercises both the explicit-selection
    path and confirms it never falls back to a pose-guided SDXL render."""
    store = StudioStore(tmp_path)
    run = store.create("footman-zimage-v1", DESCRIPTION)
    run.concept_backend = "z_image_turbo"
    store.save(run)
    qwen = FakeQwen()
    comfy = ZImageTurboFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit("footman-zimage-v1") is True
        result = wait_for(store, "footman-zimage-v1", "awaiting_review")
    stage = result.stage("D1")
    assert stage.metrics["workflow_strategy"] == "z_image_turbo_t2i_v1"
    assert len(comfy.workflows) == 2
    assert all(workflow["1"]["class_type"] == "UNETLoader" for workflow in comfy.workflows)
    assert all(
        workflow["1"]["inputs"]["unet_name"] == "z_image_turbo_int8_convrot.safetensors"
        for workflow in comfy.workflows
    )
    candidates = [
        item
        for item in stage.evidence
        if item.media_type == "image/png" and item.metrics.get("selectable") is True
    ]
    assert candidates
    assert all(item.metrics["concept_backend"] == "z_image_turbo" for item in candidates)


def test_auto_prefers_z_image_turbo_over_qwen_when_both_are_installed(tmp_path: Path) -> None:
    """Measured on an 8 GB card, one candidate costs ~48 s on Z-Image against
    ~605 s on Qwen for output in the same stylistic family. D1 renders several
    candidates per iteration and re-renders all of them on rejection, so
    letting `auto` pick Qwen turns a minutes-long stage into an hours-long
    one. Qwen stays selectable explicitly; it is just not the default."""
    store = StudioStore(tmp_path)
    run = store.create("footman-zimage-auto-v1", DESCRIPTION)
    run.concept_backend = "auto"
    store.save(run)
    qwen = FakeQwen()
    comfy = BothGeneratorsFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit("footman-zimage-auto-v1") is True
        result = wait_for(store, "footman-zimage-auto-v1", "awaiting_review")
    assert result.stage("D1").metrics["workflow_strategy"] == "z_image_turbo_t2i_v1"


def test_auto_still_falls_back_to_qwen_when_z_image_is_not_installed(tmp_path: Path) -> None:
    """The preference is an ordering, not a hard requirement: a machine with
    only Qwen installed must still get Qwen from `auto`, not SDXL."""
    store = StudioStore(tmp_path)
    run = store.create("footman-qwen-auto-v1", DESCRIPTION)
    run.concept_backend = "auto"
    store.save(run)
    qwen = FakeQwen()
    comfy = QwenImageFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit("footman-qwen-auto-v1") is True
        result = wait_for(store, "footman-qwen-auto-v1", "awaiting_review")
    assert result.stage("D1").metrics["workflow_strategy"] == "qwen_image_2512_t2i_v1"


def test_manual_qwen_image_prompt_is_preserved_rendered_and_reviewed(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("footman-manual-qwen", DESCRIPTION)
    run.spec = footman_spec()
    run.title = run.spec.title
    run.current_stage = "D1"
    run.state = "awaiting_review"
    run.stage("D0").state = "approved"
    run.stage("D1").state = "awaiting_review"
    store.save(run)
    qwen = FakeQwen()
    comfy = QwenImageFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit_manual_qwen_image(
            "footman-manual-qwen",
            "Use natural human proportions, a broad opaque shield, and a straight readable sword.",
            seed=12345,
        ) is True
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = store.load("footman-manual-qwen")
            stage = result.stage("D1")
            if stage.state == "awaiting_review" and stage.iteration == 1 and stage.qwen_reviews:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("manual Qwen Image candidate never reached review")
    stage = result.stage("D1")
    candidate = next(item for item in stage.evidence if item.metrics.get("human_prompt") is True)
    request = next(item for item in stage.evidence if item.metrics.get("role") == "human_direct_prompt")
    assert candidate.metrics["seed"] == 12345
    assert candidate.metrics["workflow_strategy"] == "qwen_image_2512_manual_prompt_v1"
    assert "natural human proportions" in store.artifact_path(result.run_id, request.relative_path).read_text(encoding="utf-8")
    assert comfy.workflows[-1]["1"]["class_type"] == "UNETLoader"


def _awaiting_d1_run(store: StudioStore, run_id: str) -> None:
    run = store.create(run_id, DESCRIPTION)
    run.spec = footman_spec()
    run.title = run.spec.title
    run.current_stage = "D1"
    run.state = "awaiting_review"
    run.stage("D0").state = "approved"
    run.stage("D1").state = "awaiting_review"
    store.save(run)


def test_control_can_release_comfy_memory_only_while_idle(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    _awaiting_d1_run(store, "footman-memory")
    comfy = ControlFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        released, message = coordinator.release_comfy_memory("footman-memory")
    assert released is True
    assert "released" in message
    assert comfy.memory_releases == [(True, True)]
    assert "comfy_memory_released" in [item["event_type"] for item in store.read_events("footman-memory")]


def test_control_stops_active_direct_qwen_render_and_preserves_run(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    _awaiting_d1_run(store, "footman-stop")
    comfy = BlockingControlFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit_manual_qwen_image(
            "footman-stop",
            "Render a natural original footman with an opaque shield and clearly gripped sword.",
        )
        assert comfy.started.wait(timeout=2)
        accepted, message = coordinator.stop("footman-stop")
        assert accepted is True
        assert "Stop requested" in message
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            stopped = store.load("footman-stop")
            if stopped.state == "blocked":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("stopped render did not become resumable")
    assert stopped.stage("D1").state == "blocked"
    assert comfy.interrupt_calls == 1
    assert "human_stopped_job" in [item["event_type"] for item in store.read_events("footman-stop")]


def test_rejection_requires_comment_and_hash_binds_evidence(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("footman-v2", DESCRIPTION)
    stage = run.stage("D1")
    stage.state = "awaiting_review"
    image = store.run_root(run.run_id) / "D1_concept" / "candidate.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "blue").save(image)
    item = store.evidence(
        run,
        "D1",
        image,
        evidence_id="candidate-1",
        label="Candidate",
        media_type="image/png",
        metrics={"selectable": True},
    )
    store.save(run)
    with pytest.raises(ValueError, match="rejection comment"):
        store.decide(run.run_id, "D1", "reject", "", item.evidence_id)
    decided = store.decide(run.run_id, "D1", "approve", "Looks good.", item.evidence_id)
    assert decided.stage("D1").human_decisions[-1].evidence_hashes == {"candidate-1": item.sha256}


def _awaiting_stage_with_one_candidate(store: StudioStore, run_id: str, stage_id: str):
    run = store.create(run_id, DESCRIPTION) if run_id not in {item.run_id for item in store.list()} else store.load(run_id)
    stage = run.stage(stage_id)
    stage.state = "awaiting_review"
    image = store.run_root(run.run_id) / f"{stage_id}_stage" / "candidate.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "blue").save(image)
    item = store.evidence(
        run,
        stage_id,
        image,
        evidence_id=f"{stage_id.lower()}-candidate-1",
        label="Candidate",
        media_type="image/png",
        metrics={"selectable": True},
    )
    store.save(run)
    return run, item


def test_assisted_decision_records_the_current_review_without_weakening_the_gate(
    tmp_path: Path,
) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "assisted-v1", "D1")
    stage = run.stage("D1")
    stage.iteration = 1
    stage.qwen_reviews.append(
        StudioQwenReview(
            review_id="review-current",
            stage_id="D1",
            iteration=stage.iteration,
            summary="The candidate meets the locked requirements.",
            recommended_evidence_id=item.evidence_id,
            confidence=0.9,
            hard_requirements_satisfied=True,
        )
    )
    store.save(run)

    decided = store.decide(
        run.run_id,
        "D1",
        "approve",
        "Accepted the AI recommendation after review.",
        item.evidence_id,
        assisted_by_review_id="review-current",
    )

    record = decided.stage("D1").human_decisions[-1]
    assert record.assisted_by_review_id == "review-current"
    assert record.evidence_hashes == {item.evidence_id: item.sha256}
    assert record.decision == "approve"


@pytest.mark.parametrize("review_id", ["review-stale", "review-missing"])
def test_assisted_decision_rejects_stale_or_unknown_review_ids(
    tmp_path: Path, review_id: str
) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, f"assisted-{review_id}", "D1")
    stage = run.stage("D1")
    stage.iteration = 2
    stage.qwen_reviews.append(
        StudioQwenReview(
            review_id="review-stale",
            stage_id="D1",
            iteration=1,
            summary="A review from an earlier attempt.",
            recommended_evidence_id=item.evidence_id,
            confidence=0.5,
        )
    )
    store.save(run)

    with pytest.raises(ValueError, match="current stage attempt"):
        store.decide(
            run.run_id,
            "D1",
            "approve",
            "",
            item.evidence_id,
            assisted_by_review_id=review_id,
        )
    assert store.load(run.run_id).stage("D1").human_decisions == []


def test_retry_requires_no_comment_and_carries_overrides_into_pending_overrides(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "retry-v1", "D1")
    decided = store.decide(
        run.run_id, "D1", "retry", "", item.evidence_id, overrides={"seed": 42}
    )
    stage = decided.stage("D1")
    assert stage.state == "pending"
    assert stage.pending_overrides == {"seed": 42}
    assert stage.human_decisions[-1].decision == "retry"
    event_types = [entry["event_type"] for entry in store.read_events(run.run_id)]
    assert "gate_retried" in event_types


@pytest.mark.parametrize(
    "bad_overrides, message",
    [
        ({"seed": -5}, "non-negative whole number"),
        ({"seed": 1.5}, "non-negative whole number"),
        ({"seed": True}, "non-negative whole number"),  # bool is an int subclass
        ({"seed": "42"}, "non-negative whole number"),
        ({"concept_steps": 0}, "between 1 and 150"),
        ({"concept_steps": 500}, "between 1 and 150"),
        ({"concept_cfg": 0}, "greater than 0"),
        ({"concept_cfg": 99}, "greater than 0 and at most 30"),
    ],
)
def test_decide_rejects_malformed_overrides_immediately(
    tmp_path: Path, bad_overrides: dict, message: str
) -> None:
    """Regression: decide() accepted any JSON object, so {"seed": -5} was
    reported as success and only surfaced later as an asynchronously failed
    stage, far from the input that caused it. The human (or API caller) must
    get the error at the moment they submit it."""
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "bad-overrides-v1", "D1")
    with pytest.raises(ValueError, match=message):
        store.decide(run.run_id, "D1", "retry", "", item.evidence_id, overrides=bad_overrides)
    # rejected cleanly -- the stage is untouched, not left half-decided
    assert store.load(run.run_id).stage("D1").state == "awaiting_review"
    assert store.load(run.run_id).stage("D1").pending_overrides == {}


def test_decide_allows_valid_and_unknown_override_keys(tmp_path: Path) -> None:
    """Valid known keys pass, and an unknown key is deliberately allowed so
    this validator does not become a chokepoint every new stage-specific
    override has to be registered in before it can be used."""
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "ok-overrides-v1", "D1")
    decided = store.decide(
        run.run_id,
        "D1",
        "retry",
        "",
        item.evidence_id,
        overrides={"seed": 0, "concept_steps": 150, "concept_cfg": 30, "some_future_key": "x"},
    )
    assert decided.stage("D1").pending_overrides["seed"] == 0
    assert decided.stage("D1").pending_overrides["some_future_key"] == "x"


def test_edit_requires_comment_or_overrides(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "edit-v1", "D1")
    with pytest.raises(ValueError, match="comment, override values, or both"):
        store.decide(run.run_id, "D1", "edit", "", item.evidence_id)
    decided = store.decide(
        run.run_id, "D1", "edit", "", item.evidence_id, overrides={"prompt_suffix": "bigger shield"}
    )
    stage = decided.stage("D1")
    assert stage.state == "rejected"
    assert stage.pending_overrides == {"prompt_suffix": "bigger shield"}


def test_skip_requires_a_reason_and_does_not_invalidate_downstream(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "skip-v1", "D1")
    with pytest.raises(ValueError, match="skip reason"):
        store.decide(run.run_id, "D1", "skip", "", item.evidence_id)
    decided = store.decide(run.run_id, "D1", "skip", "material asset, no concept needed", item.evidence_id)
    stage = decided.stage("D1")
    assert stage.state == "skipped"
    # skip does not wipe evidence or state on later stages the way reject/retry/edit do
    assert decided.stage("D4").state == "pending"
    assert decided.stage("D4").message == "Waiting for the preceding stage."


def test_rollback_reopens_an_earlier_stage_and_invalidates_everything_after_it(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("rollback-v1", DESCRIPTION)
    run.stage("D1").state = "approved"
    store.save(run)
    run, later_item = _awaiting_stage_with_one_candidate(store, "rollback-v1", "D4")
    with pytest.raises(ValueError, match="earlier stage"):
        store.decide(run.run_id, "D4", "rollback", "actually the concept was wrong", later_item.evidence_id, target_stage_id="D7")
    decided = store.decide(
        run.run_id, "D4", "rollback", "actually the concept was wrong", None, target_stage_id="D1"
    )
    assert decided.current_stage == "D1"
    assert decided.stage("D1").state == "pending"
    assert decided.stage("D4").state == "pending"
    assert decided.stage("D4").evidence == []
    # the rollback decision itself is recorded against the stage the human was standing at
    assert decided.stage("D4").human_decisions[-1].decision == "rollback"
    assert decided.stage("D4").human_decisions[-1].target_stage_id == "D1"
    event_types = [entry["event_type"] for entry in store.read_events(run.run_id)]
    assert "gate_rolled_back" in event_types


def test_rollback_target_must_have_a_prior_decision(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "rollback-v2", "D4")
    with pytest.raises(ValueError, match="no prior decision"):
        store.decide(run.run_id, "D4", "rollback", "go back", item.evidence_id, target_stage_id="D1")


class _CorrectionRecordingQwen(FakeQwen):
    """FakeQwen without the fixed-comment assertion, so a test can drive any
    correction-carrying decision and inspect what actually reached Qwen."""

    def __init__(self) -> None:
        self.correction_calls = 0
        self.seen_comments: list[str] = []

    def concept_correction_plan(self, spec, stage, candidate_ids, comparison_board=None):
        self.correction_calls += 1
        self.seen_comments.append(stage.human_decisions[-1].comment)
        return ConceptCorrectionPlan(
            operation_id="regenerate_complete_asset",
            base_evidence_id=candidate_ids[0],
            edit_box_normalized=[0.0, 0.0, 1.0, 1.0],
            positive_prompt="A " * 45 + "corrected original footman.",
            negative_prompt="missing shield, missing sword, wrong hands",
            seeds=[303, 404],
            denoise=0.8,
            diagnosis="Applying the human correction.",
            preserve=["Right-hand sword."],
        )


def _drive_to_first_d1_review(store: StudioStore, run_id: str, qwen, executor, *, comfy=None, overrides=None):
    comfy = comfy or FakeComfy()
    store.create(run_id, DESCRIPTION, overrides)
    coordinator = StudioCoordinator(
        store,
        qwen_factory=lambda run: qwen,
        comfy_factory=lambda run: comfy,
        executor=executor,
    )
    assert coordinator.submit(run_id) is True
    wait_for(store, run_id, "awaiting_review")
    return coordinator


def test_edit_decision_reaches_the_correction_path_like_a_rejection(tmp_path: Path) -> None:
    """Regression: every 'latest human correction' lookup used to filter on
    decision == "reject" alone, so an edit's comment was silently discarded
    and the stage re-ran as an unrelated fresh generation."""
    store = StudioStore(tmp_path)
    qwen = _CorrectionRecordingQwen()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = _drive_to_first_d1_review(store, "edit-path-v1", qwen, executor)
        first = store.load("edit-path-v1")
        candidate = next(
            item
            for item in first.stage("D1").evidence
            if item.metrics.get("selectable") is True
        )
        store.decide(
            "edit-path-v1",
            "D1",
            "edit",
            "Widen the shield boss.",
            candidate.evidence_id,
        )
        coordinator.submit("edit-path-v1")
        second = wait_for(store, "edit-path-v1", "awaiting_review")
    assert qwen.correction_calls == 1, "an edit must drive the targeted correction path"
    assert qwen.seen_comments == ["Widen the shield boss."]
    assert second.stage("D1").iteration == 2
    event_types = [item["event_type"] for item in store.read_events("edit-path-v1")]
    assert "gate_edited" in event_types


def test_plain_retry_does_not_use_the_correction_path(tmp_path: Path) -> None:
    """A retry is an explicit 'no judgement, just roll again', so unlike an
    edit it must NOT be turned into a targeted correction."""
    store = StudioStore(tmp_path)
    qwen = _CorrectionRecordingQwen()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = _drive_to_first_d1_review(store, "retry-path-v1", qwen, executor)
        first = store.load("retry-path-v1")
        candidate = next(
            item
            for item in first.stage("D1").evidence
            if item.metrics.get("selectable") is True
        )
        store.decide("retry-path-v1", "D1", "retry", "", candidate.evidence_id)
        coordinator.submit("retry-path-v1")
        wait_for(store, "retry-path-v1", "awaiting_review")
    assert qwen.correction_calls == 0


def test_retry_overrides_pin_the_seed_for_exactly_one_attempt(tmp_path: Path) -> None:
    """Regression: pending_overrides was written by decide() and cleared by
    _invalidate_from(), but nothing ever read it -- a retry with {"seed": N}
    silently did nothing."""
    store = StudioStore(tmp_path)
    qwen = _CorrectionRecordingQwen()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = _drive_to_first_d1_review(store, "seed-override-v1", qwen, executor)
        first = store.load("seed-override-v1")
        candidate = next(
            item
            for item in first.stage("D1").evidence
            if item.metrics.get("selectable") is True
        )
        store.decide(
            "seed-override-v1",
            "D1",
            "retry",
            "",
            candidate.evidence_id,
            overrides={"seed": 4242},
        )
        coordinator.submit("seed-override-v1")
        second = wait_for(store, "seed-override-v1", "awaiting_review")

    stage = second.stage("D1")
    second_attempt_seeds = {
        item.metrics.get("seed")
        for item in stage.evidence
        if int(item.metrics.get("iteration", 0)) == 2 and item.metrics.get("seed") is not None
    }
    assert 4242 in second_attempt_seeds, "the pinned seed must actually reach the render"
    # consumed exactly once, so a later automatic iteration is not silently re-pinned
    assert stage.pending_overrides == {}
    applied = [
        item
        for item in store.read_events("seed-override-v1")
        if item["event_type"] == "stage_overrides_applied"
    ]
    assert len(applied) == 1
    assert applied[0]["payload"]["overrides"] == {"seed": 4242}


def test_a_run_configured_with_high_quality_actually_submits_high_quality_workflows(
    tmp_path: Path,
) -> None:
    """End-to-end proof that [quality.high]'s concept_steps/concept_cfg reach
    the real KSampler node ComfyUI would receive, not just that the numbers
    parse. FakeComfy.generate() never touches a GPU; it only records what
    was submitted, exactly like the seed-override regression test above."""
    store = StudioStore(tmp_path)
    comfy = FakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        _drive_to_first_d1_review(
            store,
            "quality-high-v1",
            FakeQwen(),
            executor,
            comfy=comfy,
            overrides={"concept_steps": 45, "concept_cfg": 7.0},
        )
    assert comfy.workflows, "D1 must have submitted at least one workflow"
    # Only concept_workflow() outputs carry quality-tier steps/cfg; the
    # footman spec's deferred shield repair also submits an inpaint_workflow()
    # KSampler with its own fixed steps/cfg, distinguishable by EmptyLatentImage
    # (concept_workflow always creates a fresh latent; inpaint never does).
    concept_workflows = [
        workflow
        for workflow in comfy.workflows
        if any(node["class_type"] == "EmptyLatentImage" for node in workflow.values())
    ]
    assert concept_workflows, "no concept_workflow() output was submitted"
    samplers = [
        node for workflow in concept_workflows for node in workflow.values() if node["class_type"] == "KSampler"
    ]
    assert samplers, "no KSampler node was submitted"
    assert all(node["inputs"]["steps"] == 45 for node in samplers)
    assert all(node["inputs"]["cfg"] == 7.0 for node in samplers)


def test_a_run_with_no_quality_override_submits_todays_real_default_workflow(tmp_path: Path) -> None:
    """Companion to the test above: a run with no quality override at all
    must submit exactly the steps=30/cfg=6.0 that concept_workflow() has
    always defaulted to -- proving the new quality-tier wiring introduced
    no behavior change for the common case."""
    store = StudioStore(tmp_path)
    comfy = FakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        _drive_to_first_d1_review(store, "quality-default-v1", FakeQwen(), executor, comfy=comfy)
    concept_workflows = [
        workflow
        for workflow in comfy.workflows
        if any(node["class_type"] == "EmptyLatentImage" for node in workflow.values())
    ]
    assert concept_workflows, "no concept_workflow() output was submitted"
    samplers = [
        node for workflow in concept_workflows for node in workflow.values() if node["class_type"] == "KSampler"
    ]
    assert samplers
    assert all(node["inputs"]["steps"] == 30 for node in samplers)
    assert all(node["inputs"]["cfg"] == 6.0 for node in samplers)


def test_d2_geometry_generation_runs_end_to_end_through_a_fake_worker(tmp_path: Path) -> None:
    """First real pipeline-level test of D2, which has never been driven
    through the coordinator before -- it always stopped at D1's gate. Proves
    the worker_executor injection seam works and that _run_d2's real
    evidence/gate logic (vertices/faces/watertight from an actually-parsed
    GLB, hard_gate_passed, Qwen review) reaches 'approved', not just that a
    canned response satisfies a mock.

    Takes no config.local.toml monkeypatch: D2's blob-path computation now
    goes through _workspace_relative() and is based on the store's own
    workspace, so a run driven by an injected worker_executor needs no
    machine config at all.
    """
    store = StudioStore(tmp_path)
    qwen = FakeQwen()
    comfy = FakeComfy()
    worker = FakeWorkerExecutor()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: qwen,
            comfy_factory=lambda run: comfy,
            worker_executor=worker,
            executor=executor,
        )
        store.create("d2-e2e-v1", DESCRIPTION)
        assert coordinator.submit("d2-e2e-v1") is True
        first = wait_for(store, "d2-e2e-v1", "awaiting_review")
        assert first.current_stage == "D1"
        candidate = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("d2-e2e-v1", "D1", "approve", "Looks good.", candidate.evidence_id)
        coordinator.submit("d2-e2e-v1")
        # D2 has no human gate, so the run never parks in awaiting_review for it.
        run = _wait_for_stage_settled(store, "d2-e2e-v1", "D2")

    d2 = run.stage("D2")
    assert d2.state == "approved", d2.error
    assert d2.metrics["vertices"] == 2562  # trimesh.creation.icosphere(subdivisions=4), computed for real
    assert d2.metrics["faces"] == 5120
    assert d2.metrics["watertight"] is True
    assert d2.metrics["hard_gate_passed"] is True
    assert d2.metrics["backend"] == "hunyuan3d"
    # The default backend is native ComfyUI, so D2 itself runs no subprocess
    # worker. Later stages (D3 onward) legitimately still do.
    assert not any(item.operation_id == "geometry.generate_from_rgba" for item in worker.requests)
    glb_evidence = [item for item in d2.evidence if item.media_type == "model/gltf-binary"]
    assert len(glb_evidence) == 1


def test_d2_generates_the_mesh_from_the_image_the_human_approved(tmp_path: Path) -> None:
    """The whole text -> 2D -> 3D chain, and the bug that used to break it.

    D2 previously fed the approved concept to Qwen only as context for a text
    prompt, rendered a BRAND NEW image from that prompt, and sent that to the
    3D model. So the mesh came from an image nobody had reviewed, and
    approving a concept had no effect on the geometry. D2 must now key the
    approved image itself to RGBA and hand exactly that to image-to-3D.
    """
    store = StudioStore(tmp_path)
    comfy = FakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: comfy,
            worker_executor=FakeWorkerExecutor(),
            executor=executor,
        )
        store.create("d2-chain-v1", DESCRIPTION)
        coordinator.submit("d2-chain-v1")
        first = wait_for(store, "d2-chain-v1", "awaiting_review")
        approved = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("d2-chain-v1", "D1", "approve", "Looks good.", approved.evidence_id)
        coordinator.submit("d2-chain-v1")
        run = _wait_for_stage_settled(store, "d2-chain-v1", "D2")

    d2 = run.stage("D2")
    assert d2.state == "approved", d2.error

    rgba = next(item for item in d2.evidence if item.evidence_id.endswith("rgba-seed"))
    assert rgba.metrics["source_evidence"] == "d1_approved_concept"
    assert rgba.metrics["source_sha256"] == approved.sha256

    # Compare actual pixels, not just the recorded provenance. A metric can be
    # mislabelled -- an earlier version of this test asserted only on
    # source_sha256 and passed even when D2 was keying a freshly rendered
    # image, because that field was populated from the approved concept
    # regardless of what was really used. make_chroma_alpha is deterministic,
    # so re-keying the approved concept here must reproduce D2's RGBA byte for
    # byte; keying anything else cannot.
    from text2model_forge.studio_comfy import make_chroma_alpha as _key

    expected = tmp_path / "expected_rgba.png"
    _key(store.artifact_path("d2-chain-v1", approved.relative_path), expected)
    actual = store.artifact_path("d2-chain-v1", rgba.relative_path)
    assert actual.read_bytes() == expected.read_bytes(), (
        "D2's RGBA input does not match a chroma-key of the approved concept, "
        "so the mesh derives from some other image"
    )

    # D2 submitted an image-to-3D graph, not another text-to-image one.
    mesh_workflows = [
        w for w in comfy.workflows if any(n["class_type"] == "SaveGLB" for n in w.values())
    ]
    assert len(mesh_workflows) == 1, "D2 must submit exactly one image-to-3D workflow"
    classes = {n["class_type"] for n in mesh_workflows[0].values()}
    assert "LoadImage" in classes, "the workflow must consume an uploaded image"
    assert "Hunyuan3Dv2Conditioning" in classes
    assert "EmptyLatentImage" not in classes, "an image-to-3D graph must not start from an empty latent"


def test_d2_backend_is_configurable_back_to_a_subprocess_worker(tmp_path: Path) -> None:
    """Machines with the VRAM for TRELLIS.2 can still route D2 through the
    typed worker protocol by setting [stages.D2].backend."""
    store = StudioStore(tmp_path)
    worker = FakeWorkerExecutor()
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "base.toml").write_text(
        '[stages.D2]\nbackend = "trellis2.4b"\nworker_id = "trellis2.4b"\n', encoding="utf-8"
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=worker,
            executor=executor,
        )
        coordinator._stage_settings = lambda run, stage_id: (
            {"backend": "trellis2.4b", "worker_id": "trellis2.4b"} if stage_id == "D2" else {}
        )
        store.create("d2-backend-v1", DESCRIPTION)
        coordinator.submit("d2-backend-v1")
        first = wait_for(store, "d2-backend-v1", "awaiting_review")
        approved = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("d2-backend-v1", "D1", "approve", "Looks good.", approved.evidence_id)
        coordinator.submit("d2-backend-v1")
        run = _wait_for_stage_settled(store, "d2-backend-v1", "D2")

    assert run.stage("D2").state == "approved", run.stage("D2").error
    geometry = [item for item in worker.requests if item.operation_id == "geometry.generate_from_rgba"]
    assert len(geometry) == 1, "D2 must have routed through the subprocess worker"
    # It still receives the approved concept, keyed to RGBA -- only the
    # backend changed, not where the image comes from.
    assert list(geometry[0].input_paths.values())[0].endswith("geometry_seed_rgba.png")


def test_automatic_oom_retry_is_bounded_and_changes_the_memory_plan(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("oom-retry-v1", DESCRIPTION)
    run.current_stage = "D1"
    run.state = "running"
    run.stage("D1").state = "running"
    store.save(run)
    coordinator = StudioCoordinator(
        store,
        qwen_factory=lambda current: FakeQwen(),
        comfy_factory=lambda current: FakeComfy(),
    )
    try:
        assert coordinator._schedule_automatic_retry(
            "oom-retry-v1", RuntimeError("CUDA out of memory")
        ) is True
        retried = store.load("oom-retry-v1").stage("D1")
        assert retried.retry_attempt == 1
        assert retried.pending_overrides == {
            "concept_width": 576,
            "concept_height": 768,
            "concept_candidates": 2,
        }
        assert coordinator._schedule_automatic_retry(
            "oom-retry-v1", RuntimeError("GPU admission denied")
        ) is True
        second = store.load("oom-retry-v1").stage("D1")
        assert second.pending_overrides == {
            "concept_width": 512,
            "concept_height": 640,
            "concept_candidates": 2,
        }
        assert coordinator._schedule_automatic_retry(
            "oom-retry-v1", RuntimeError("CUDA out of memory")
        ) is False
    finally:
        coordinator.close()


def test_d3_cleanup_runs_end_to_end_through_the_same_fake_worker(tmp_path: Path) -> None:
    """Validates the README's claim that D3 is 'one FakeQwen method away'
    from the treatment D2 got: FakeQwen.review_cleanup and
    FakeWorkerExecutor's blender.repair fixture already existed, so this
    needed no new production code and no new fake -- only a test. D3 runs
    the real Blender-worker request path (via the injected executor), builds
    a real image board from the returned PNG, and applies its real Qwen
    identity gate."""
    store = StudioStore(tmp_path)
    worker = FakeWorkerExecutor()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=worker,
            executor=executor,
        )
        store.create("d3-e2e-v1", DESCRIPTION)
        coordinator.submit("d3-e2e-v1")
        first = wait_for(store, "d3-e2e-v1", "awaiting_review")
        candidate = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("d3-e2e-v1", "D1", "approve", "Looks good.", candidate.evidence_id)
        coordinator.submit("d3-e2e-v1")
        run = _wait_for_stage_settled(store, "d3-e2e-v1", "D3")

    d3 = run.stage("D3")
    assert d3.state == "approved", d3.error
    assert run.stage("D2").state == "approved"
    repair_requests = [item for item in worker.requests if item.operation_id == "blender.repair"]
    assert len(repair_requests) == 1
    # A deformable character must ask Blender for a single connected component.
    assert repair_requests[0].parameters["component_policy"] == "keep_largest"
    assert repair_requests[0].parameters["maximum_connected_components"] == 1
    # D3 must carry forward a usable cleaned mesh for D4.
    assert any(
        item.media_type == "model/gltf-binary" and item.metrics.get("role") == "candidate_geometry"
        for item in d3.evidence
    )


def _approve_gate(store: StudioStore, coordinator, run_id: str, stage_id: str, comment: str = "Looks good."):
    """Approve one human gate, picking the stage's selectable candidate."""
    run = wait_for(store, run_id, "awaiting_review")
    assert run.current_stage == stage_id, f"expected {stage_id}, got {run.current_stage}"
    candidate = next(
        item for item in run.stage(stage_id).evidence if item.metrics.get("selectable") is True
    )
    store.decide(run_id, stage_id, "approve", comment, candidate.evidence_id)
    coordinator.submit(run_id)


def test_d4_through_d6_run_end_to_end_for_a_deformable_character(tmp_path: Path) -> None:
    """D4 (rig proposal) is a human gate driven by a real Blender-worker
    request; D5 and D6 run no worker of their own and instead adopt D4's
    hash-bound outputs via _adopt_d4_output, which reads each adopted report's
    gate field. That adoption contract is what this proves -- the fake must
    emit exactly the roles those two stages require, so a missing role fails
    here the same way a real misbehaving worker would."""
    store = StudioStore(tmp_path)
    worker = FakeWorkerExecutor()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=worker,
            executor=executor,
        )
        store.create("d4-e2e-v1", DESCRIPTION)
        coordinator.submit("d4-e2e-v1")
        _approve_gate(store, coordinator, "d4-e2e-v1", "D1")
        _approve_gate(store, coordinator, "d4-e2e-v1", "D4")
        run = _wait_for_stage_settled(store, "d4-e2e-v1", "D6")

    assert run.stage("D4").state == "approved", run.stage("D4").error
    assert run.stage("D5").state == "approved", run.stage("D5").error
    assert run.stage("D6").state == "approved", run.stage("D6").error
    rig_requests = [
        item for item in worker.requests if item.operation_id == "blender.propose_short_biped_rig"
    ]
    assert len(rig_requests) == 1
    # D5/D6 adopt from D4 rather than running their own worker.
    assert run.stage("D5").metrics["source_stage"] == "D4"
    assert run.stage("D6").metrics["source_stage"] == "D4"
    assert run.stage("D5").metrics["hard_gate_passed"] is True
    assert run.stage("D6").metrics["adopted_artifacts"] >= 1


def test_d6_adoption_fails_closed_when_an_adopted_report_did_not_pass_its_gate(
    tmp_path: Path,
) -> None:
    """_adopt_d4_output reads gate_passed out of each adopted JSON report and
    must refuse to approve when one is false -- otherwise a failed skinning
    check would be silently promoted into an approved stage."""

    class FailedSkinningWorker(FakeWorkerExecutor):
        def _op_blender_propose_short_biped_rig(self, request, output_root):
            outputs, diagnostics = super()._op_blender_propose_short_biped_rig(request, output_root)
            for output in outputs:
                if output.role == "skinning_report":
                    Path(output.path).write_text(
                        json.dumps({"gate_passed": False, "reason": "collapsed shoulder"}),
                        encoding="utf-8",
                    )
            return outputs, diagnostics

    store = StudioStore(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=FailedSkinningWorker(),
            executor=executor,
        )
        store.create("d6-gate-v1", DESCRIPTION)
        coordinator.submit("d6-gate-v1")
        _approve_gate(store, coordinator, "d6-gate-v1", "D1")
        _approve_gate(store, coordinator, "d6-gate-v1", "D4")
        run = _wait_for_stage_settled(store, "d6-gate-v1", "D6")

    d6 = run.stage("D6")
    assert d6.state == "failed"
    assert "hard gate" in (d6.error or "")
    # D5 adopts a different role set, so it is unaffected by the skinning failure.
    assert run.stage("D5").state == "approved"


def test_a_deformable_character_runs_the_whole_chain_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full character path, the counterpart to the static-prop chain.
    Unlike a prop this runs every stage, and D7/D9/D10 all drive the
    resumable motion chain (adapters/run_motion_candidate_pipeline.py) via
    the script_runner seam with progressively later --stop-after points.

    Four human gates: D1 concept, D4 rig, D7 motion, D8 surface -- then D10
    parks awaiting the final ship approval, which is the correct successful
    end of a run rather than an auto-approval."""
    monkeypatch.setattr(
        "text2model_forge.studio_pipeline.load_local_config",
        lambda *a, **k: _machine_config_with_blender(tmp_path),
    )
    store = StudioStore(tmp_path)
    scripts = FakeScriptRunner()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=FakeWorkerExecutor(),
            script_runner=scripts,
            executor=executor,
        )
        store.create("character-chain-v1", DESCRIPTION)
        coordinator.submit("character-chain-v1")
        for gate in ("D1", "D4", "D7", "D8"):
            _approve_gate(store, coordinator, "character-chain-v1", gate)
        run = _wait_for_stage_settled(store, "character-chain-v1", "D10")

    for stage_id in ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"):
        assert run.stage(stage_id).state == "approved", f"{stage_id}: {run.stage(stage_id).error}"
    assert run.stage("D10").state == "awaiting_review", run.stage("D10").error
    # Nothing is skipped for a deformable character -- the exact opposite of the prop.
    assert not any(item.state == "skipped" for item in run.stages)
    # The motion chain was resumed at progressively later stop points.
    stop_points = [
        cmd[cmd.index("--stop-after") + 1]
        for cmd in scripts.commands
        if "--stop-after" in cmd
    ]
    assert stop_points == ["retarget_qwen_review", "sprite_qwen_review", "unity_smoke_bundle"]


CHAIR_DESCRIPTION = (
    "An original sturdy oak dining chair with a straight slatted back and four tapered legs."
)


def chair_spec() -> StudioAssetSpec:
    return StudioAssetSpec(
        asset_id="oak_chair",
        title="Original Oak Chair",
        description=CHAIR_DESCRIPTION,
        creative_direction="Readable original cottage furniture, no franchise motifs.",
        asset_kind="prop",
        behavior="static",
        anatomy_family=None,
        height_m=0.9,
        dimensions_m=[0.45, 0.9, 0.45],
        silhouette=["straight slatted back", "four tapered legs"],
        materials=["oak wood"],
        animations=[],
        locked_features=["four legs", "slatted back"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["silhouette reads at prop scale"],
    )


class ChairQwen(FakeQwen):
    """FakeQwen for a static prop: compiles the chair spec and reviews its
    surface, with none of the character-specific equipment handling."""

    def compile_spec(self, description):
        return chair_spec()

    def concept_plan(self, spec, stage):
        return ConceptPlan(
            positive_prompt="A " * 45 + "original oak dining chair, straight slatted back, four legs.",
            negative_prompt="people, characters, weapons, copied design",
            seeds=[701 + stage.iteration, 802 + stage.iteration],
            rationale="Two deterministic seeds for a static prop.",
        )

    def revision_plan(self, spec, stage):
        from text2model_forge.studio_qwen import RevisionPlan

        return RevisionPlan(
            diagnosis="Surface reads acceptably at prop scale.",
            changes=["No further change required."],
            preserve=["Leg taper."],
        )


# _run_motion_chain requires this exact CC0 clip under the configured
# workspace_root before it will retarget anything. The path is hardcoded in
# studio_pipeline.py rather than resolved through src/text2model_forge/motion_library.py
# -- see the motion-library gap noted in the README.
_MOTION_SOURCE_RELATIVE = Path(
    "sources"
) / "quaternius_universal_animation_library_standard" / "Universal Animation Library[Standard]" / "Unreal-Godot" / "UAL1_Standard.glb"


def _machine_config_with_blender(tmp_path: Path) -> Text2ModelLocalConfig:
    """A realistic machine config for tests that reach D7/D8/D9.

    D8 resolves the configured Blender executable, and D7's motion chain
    requires the qualified CC0 motion clip to exist. Both checks are correct
    production behavior and are deliberately NOT bypassed here; the test
    supplies real (if inert) files so the resolution genuinely succeeds
    rather than being stubbed out."""
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"inert-blender-stand-in")
    motion_source = tmp_path / _MOTION_SOURCE_RELATIVE
    motion_source.parent.mkdir(parents=True, exist_ok=True)
    _synthetic_mesh(motion_source)
    return Text2ModelLocalConfig(
        workspace_root=str(tmp_path),
        workers={"blender": WorkerBinding(command_prefix=[str(blender)], environment={})},
    )


def test_a_static_prop_runs_the_whole_chain_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 'ask for a chair, get a chair' path, as far as it can be driven
    without a GPU. A static prop skips D4-D7 entirely (no skeleton, rig,
    skin, or motion), so the real chain is D0 -> D1 -> D2 -> D3 -> D8 -> D9
    -> D10. This exercises both subprocess boundaries at once: the typed
    worker protocol (D2/D3/D9) through worker_executor, and the adapters/
    helper scripts (D8's bake and review) through script_runner.

    Until now the prop path was only proven at D0 -- that its stage-skip
    logic computed the right states. This proves the stages that remain
    actually run for a non-character asset."""
    monkeypatch.setattr(
        "text2model_forge.studio_pipeline.load_local_config",
        lambda *a, **k: _machine_config_with_blender(tmp_path),
    )
    store = StudioStore(tmp_path)
    worker = FakeWorkerExecutor()
    scripts = FakeScriptRunner()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: ChairQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=worker,
            script_runner=scripts,
            executor=executor,
        )
        store.create("chair-v1", CHAIR_DESCRIPTION)
        coordinator.submit("chair-v1")
        _approve_gate(store, coordinator, "chair-v1", "D1")
        _approve_gate(store, coordinator, "chair-v1", "D8")
        run = _wait_for_stage_settled(store, "chair-v1", "D10")

    assert run.spec.asset_kind == "prop"
    # The articulation stages are skipped, not run and not failed.
    for stage_id in ("D4", "D5", "D6", "D7"):
        assert run.stage(stage_id).state == "skipped", stage_id
    # Everything a static prop genuinely needs actually ran.
    for stage_id in ("D0", "D1", "D2", "D3", "D8", "D9"):
        assert run.stage(stage_id).state == "approved", f"{stage_id}: {run.stage(stage_id).error}"
    assert run.stage("D10").state == "awaiting_review", run.stage("D10").error

    # D9 rendered a real delivery board and hash-bound manifest for the prop.
    assert run.stage("D9").metrics["automatic_gate_passed"] is True
    assert any(item.metrics.get("role") == "delivery_manifest" for item in run.stage("D9").evidence)
    # D10 built a runtime manifest that still demands human approval.
    manifest_item = next(
        item for item in run.stage("D10").evidence if item.media_type == "application/json"
    )
    manifest = json.loads(
        store.artifact_path("chair-v1", manifest_item.relative_path).read_text(encoding="utf-8")
    )
    assert manifest["asset_kind"] == "prop"
    assert manifest["human_approved"] is False
    assert manifest["human_approval_required"] is True
    # The prop never asked for a rig, and D8's bake really did run.
    assert not any(
        item.operation_id == "blender.propose_short_biped_rig" for item in worker.requests
    )
    assert any("bake_surface.py" in " ".join(cmd) for cmd in scripts.commands)


def test_stage_config_reaches_the_real_worker_requests(tmp_path: Path) -> None:
    """Config-wiring proof: [stages.*] values must actually appear in the
    typed worker requests, not merely resolve. Uses the advanced profile,
    which raises D3/D4's render_size above base.

    D2 is deliberately not checked here: on the default hunyuan3d backend it
    goes through ComfyUI rather than the worker protocol, and its own config
    is asserted in the D2 image-to-3D tests instead.
    """
    store = StudioStore(tmp_path)
    worker = FakeWorkerExecutor()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=worker,
            executor=executor,
        )
        store.create("stage-config-v1", DESCRIPTION, {"profile": "advanced"})
        coordinator.submit("stage-config-v1")
        _approve_gate(store, coordinator, "stage-config-v1", "D1")
        _wait_for_stage_settled(store, "stage-config-v1", "D4")

    by_operation = {item.operation_id: item for item in worker.requests}
    # base render_size is 512; advanced raises D3 and D4 to 768.
    assert by_operation["blender.repair"].parameters["render_size"] == 768
    assert by_operation["blender.propose_short_biped_rig"].parameters["render_size"] == 768
    # A key only base.toml sets still flows through.
    assert by_operation["blender.propose_short_biped_rig"].parameters["maximum_bone_influences"] == 4


def test_a_run_keeps_its_own_profile_rather_than_the_current_default(tmp_path: Path) -> None:
    """StudioRun.profile pins which profile a run resolves from, so two runs
    with different profiles get different stage parameters concurrently."""
    store = StudioStore(tmp_path)
    default_run = store.create("profile-default-v1", DESCRIPTION)
    advanced_run = store.create("profile-advanced-v1", DESCRIPTION, {"profile": "advanced"})
    assert default_run.profile == "simple"
    assert advanced_run.profile == "advanced"

    coordinator = StudioCoordinator(store, qwen_factory=lambda run: FakeQwen())
    assert coordinator._stage_settings(default_run, "D2")["texture_size"] == 2048
    assert coordinator._stage_settings(advanced_run, "D2")["texture_size"] == 4096


def test_stage_settings_fall_back_to_defaults_for_an_unknown_profile(tmp_path: Path) -> None:
    """A run naming a profile that no longer exists must fall back to base
    rather than fail -- configuration should not brick an in-flight run."""
    store = StudioStore(tmp_path)
    run = store.create("missing-profile-v1", DESCRIPTION, {"profile": "deleted-profile"})
    coordinator = StudioCoordinator(store, qwen_factory=lambda run: FakeQwen())
    settings = coordinator._stage_settings(run, "D2")
    assert settings["texture_size"] == 2048  # base.toml's value
    assert coordinator._stage_settings(run, "D99") == {}  # unknown stage is empty, not an error


def test_d7_uses_the_historical_default_clip_when_no_catalog_id_is_set(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    coordinator = StudioCoordinator(store, qwen_factory=lambda run: FakeQwen())
    run = store.create("donor-default-v1", DESCRIPTION)
    _machine_config_with_blender(tmp_path)  # creates the default clip on disk
    resolved = coordinator._resolve_donor_motion(run, tmp_path)
    assert resolved == (tmp_path / _MOTION_SOURCE_RELATIVE).resolve()


def test_d7_resolves_a_configured_donor_motion_id_through_the_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The motion catalog was previously unused by the pipeline: base.toml
    documented donor_motion_id but _run_motion_chain hardcoded one clip and
    never consulted it."""
    catalog_dir = tmp_path / "motion_library"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    clip = _synthetic_mesh(catalog_dir / "clips" / "sprint.glb")
    (catalog_dir / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clips": [
                    {
                        "clip_id": "sprint_v1",
                        "display_name": "Sprint",
                        "description": "A CC0 sprint cycle.",
                        "compatible_anatomy_family": "humanoid",
                        "source_url": "https://example.com/sprint",
                        "author": "Example Author",
                        "license": "CC0-1.0",
                        "license_requires_attribution": False,
                        "local_path": "clips/sprint.glb",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    store = StudioStore(tmp_path)
    coordinator = StudioCoordinator(store, qwen_factory=lambda run: FakeQwen())
    run = store.create("donor-catalog-v1", DESCRIPTION)
    monkeypatch.setattr(
        coordinator,
        "_stage_settings",
        lambda run, stage_id: {"donor_motion_id": "sprint_v1"} if stage_id == "D7" else {},
    )
    assert coordinator._resolve_donor_motion(run, tmp_path) == clip.resolve()


def test_d7_fails_loudly_for_an_unknown_donor_motion_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrecognized id must never silently fall back to the default clip --
    that would retarget a motion the operator did not choose."""
    catalog_dir = tmp_path / "motion_library"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "catalog.json").write_text(
        json.dumps({"schema_version": 1, "clips": []}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    store = StudioStore(tmp_path)
    coordinator = StudioCoordinator(store, qwen_factory=lambda run: FakeQwen())
    run = store.create("donor-unknown-v1", DESCRIPTION)
    monkeypatch.setattr(
        coordinator,
        "_stage_settings",
        lambda run, stage_id: {"donor_motion_id": "does_not_exist"} if stage_id == "D7" else {},
    )
    with pytest.raises(ValueError, match="unknown donor_motion_id"):
        coordinator._resolve_donor_motion(run, tmp_path)


def test_d2_survives_a_studio_workspace_that_differs_from_config_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `text2model_forge studio --workspace X` explicitly allows X to
    differ from config.local.toml's workspace_root (see cli.py's
    `args.workspace or ...`), but D2 computed its artifact blob_path as
    rgba_seed.relative_to(config.workspace_root). When they diverged that
    raised an opaque ValueError -- "'...' is not in the subpath of '...'" --
    and failed the stage. The base must be the store's own workspace, which
    every Studio-run file lives under by construction."""
    studio_workspace = tmp_path / "studio_workspace"
    config_workspace = tmp_path / "config_workspace"
    config_workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "text2model_forge.studio_pipeline.load_local_config",
        lambda *a, **k: Text2ModelLocalConfig(workspace_root=str(config_workspace), workers={}),
    )
    store = StudioStore(studio_workspace)
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=FakeWorkerExecutor(),
            executor=executor,
        )
        store.create("diverging-ws-v1", DESCRIPTION)
        coordinator.submit("diverging-ws-v1")
        first = wait_for(store, "diverging-ws-v1", "awaiting_review")
        candidate = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("diverging-ws-v1", "D1", "approve", "Looks good.", candidate.evidence_id)
        coordinator.submit("diverging-ws-v1")
        run = _wait_for_stage_settled(store, "diverging-ws-v1", "D2")
    assert run.stage("D2").state == "approved", run.stage("D2").error


def test_d2_needs_no_machine_config_when_a_worker_executor_is_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: D2 was the one load_local_config() call site of five with
    no None guard, so running without config.local.toml crashed with
    AttributeError: 'NoneType' object has no attribute 'workspace_root'
    instead of the clear RuntimeError every other stage raises. The blob-path
    computation no longer consults machine config at all, so this path is
    simply gone rather than merely better-worded."""
    monkeypatch.setattr("text2model_forge.studio_pipeline.load_local_config", lambda *a, **k: None)
    store = StudioStore(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=FakeWorkerExecutor(),
            executor=executor,
        )
        store.create("no-machine-config-v1", DESCRIPTION)
        coordinator.submit("no-machine-config-v1")
        first = wait_for(store, "no-machine-config-v1", "awaiting_review")
        candidate = next(
            item for item in first.stage("D1").evidence if item.metrics.get("selectable") is True
        )
        store.decide("no-machine-config-v1", "D1", "approve", "Looks good.", candidate.evidence_id)
        coordinator.submit("no-machine-config-v1")
        run = _wait_for_stage_settled(store, "no-machine-config-v1", "D2")
    assert run.stage("D2").state == "approved", run.stage("D2").error


def test_save_retries_a_transient_windows_permission_error_on_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for an intermittently-failing test suite: a real run seen
    while building the D2 worker-executor seam (test_pipeline_prefers_
    installed_qwen_image_generation..., no relation to D2 itself) failed with
    PermissionError: [WinError 5] Access is denied on run.json.tmp -> run.json,
    then passed cleanly on rerun. Path.replace() is atomic on POSIX but can
    transiently fail on Windows when another reader briefly has the file
    open; save() must absorb a few such failures rather than fail the whole
    stage over a race that clears in milliseconds."""
    from pathlib import Path as PathType

    store = StudioStore(tmp_path)
    run = store.create("flaky-save-v1", DESCRIPTION)

    real_replace = PathType.replace
    calls = {"count": 0}

    def flaky_replace(self, target):
        calls["count"] += 1
        if calls["count"] <= 2 and self.name == "run.json.tmp":
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(PathType, "replace", flaky_replace)
    store.save(run)  # must not raise, despite the first two attempts failing
    assert calls["count"] == 3
    assert store.load("flaky-save-v1").run_id == "flaky-save-v1"


def test_save_gives_up_after_persistent_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry is bounded -- a real, non-transient lock must still surface
    as an error rather than hang or silently drop the write."""
    from pathlib import Path as PathType

    store = StudioStore(tmp_path)
    run = store.create("stuck-save-v1", DESCRIPTION)

    def always_fails(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(PathType, "replace", always_fails)
    with pytest.raises(PermissionError):
        store.save(run)


def test_two_store_instances_reject_a_stale_save_instead_of_clobbering_it(
    tmp_path: Path,
) -> None:
    first = StudioStore(tmp_path)
    second = StudioStore(tmp_path)
    first.create("shared-workspace-v1", DESCRIPTION)

    first_view = first.load("shared-workspace-v1")
    stale_second_view = second.load("shared-workspace-v1")
    first_view.title = "Saved by the browser"
    first.save(first_view)

    stale_second_view.title = "Stale CLI overwrite"
    with pytest.raises(StudioConflictError, match="changed after it was loaded"):
        second.save(stale_second_view)

    persisted = first.load("shared-workspace-v1")
    assert persisted.title == "Saved by the browser"
    assert persisted.revision == first_view.revision


def test_stale_event_is_refused_before_it_is_appended(tmp_path: Path) -> None:
    first = StudioStore(tmp_path)
    second = StudioStore(tmp_path)
    first.create("shared-events-v1", DESCRIPTION)
    current = first.load("shared-events-v1")
    stale = second.load("shared-events-v1")
    current.title = "Current"
    first.save(current)
    before = first.read_events("shared-events-v1")

    with pytest.raises(StudioConflictError, match="before event"):
        second.event(stale, "stale_event", {})

    assert first.read_events("shared-events-v1") == before


def test_artifact_paths_cannot_escape_run(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    store.create("footman-v3", DESCRIPTION)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        store.artifact_path("footman-v3", "../../../outside.txt")


def test_invalidating_work_does_not_reopen_a_contract_skipped_stage(tmp_path: Path) -> None:
    """Regression: _invalidate_from() reset every downstream stage to pending
    unconditionally, including the ones D0's compiled contract had marked not
    applicable. Rejecting the concept of a static prop therefore un-skipped
    D4-D7 and sent a chair through skeleton, rig, skinning, and motion."""
    store = StudioStore(tmp_path)
    run = store.create("static-prop-reject", DESCRIPTION)
    run.stage("D0").state = "approved"
    for stage_id in ("D4", "D5", "D6", "D7"):
        stage = run.stage(stage_id)
        stage.state = "skipped"
        stage.applicable = False
        stage.progress = 1
        stage.message = "Static assets do not require skeleton, rig, deformation, or motion."
    store.save(run)
    run, item = _awaiting_stage_with_one_candidate(store, "static-prop-reject", "D1")

    decided = store.decide(run.run_id, "D1", "reject", "Legs read as too thin.", item.evidence_id)

    for stage_id in ("D4", "D5", "D6", "D7"):
        stage = decided.stage(stage_id)
        assert stage.state == "skipped", f"{stage_id} was reopened by an upstream rejection"
        assert stage.applicable is False
        assert stage.message.startswith("Static assets"), "the skip reason was overwritten"
    # ...while the stages that do apply are invalidated exactly as before.
    assert decided.stage("D2").state == "pending"
    assert decided.stage("D8").state == "pending"
    assert StudioCoordinator._next_stage(decided).stage_id == "D1"


def test_rollback_past_a_contract_skipped_stage_leaves_it_skipped(tmp_path: Path) -> None:
    """The same defect through the rollback path: a rollback to D2 must not
    schedule the rig stages a static asset's contract already excluded."""
    store = StudioStore(tmp_path)
    run = store.create("static-prop-rollback", DESCRIPTION)
    for stage_id in ("D0", "D1", "D2", "D3"):
        run.stage(stage_id).state = "approved"
    for stage_id in ("D4", "D5", "D6", "D7"):
        run.stage(stage_id).state = "skipped"
        run.stage(stage_id).applicable = False
    store.save(run)
    run, item = _awaiting_stage_with_one_candidate(store, "static-prop-rollback", "D8")

    decided = store.decide(
        run.run_id, "D8", "rollback", "the cleanup lost a leg", item.evidence_id, target_stage_id="D2"
    )

    assert decided.current_stage == "D2"
    assert [stage.state for stage in decided.stages[4:8]] == ["skipped"] * 4
    # D3 -> D8 directly, never through the excluded articulation stages.
    decided.stage("D2").state = "approved"
    decided.stage("D3").state = "approved"
    assert StudioCoordinator._next_stage(decided).stage_id == "D8"


def test_recompiling_d0_reopens_a_previously_skipped_stage(tmp_path: Path) -> None:
    """Because _invalidate_from() now preserves contract skips, D0 is the only
    thing that can lift one -- so re-running it against a differently shaped
    spec must reset applicability rather than inherit the old contract's."""
    chair = StudioAssetSpec(
        asset_id="oak_chair",
        title="Original Oak Chair",
        description="A simple original sturdy wooden dining chair with a straight back.",
        creative_direction="Readable original cottage-style furniture.",
        asset_kind="prop",
        behavior="static",
        anatomy_family=None,
        height_m=0.9,
        dimensions_m=[0.45, 0.9, 0.45],
        silhouette=["straight back"],
        materials=["oak wood"],
        animations=[],
        locked_features=["four legs"],
        negative_constraints=["no copied franchise motifs"],
        gameplay_readability=["silhouette reads at prop scale"],
    )
    store = StudioStore(tmp_path)
    run = store.create("recompiled-v1", chair.description)
    coordinator = StudioCoordinator(store, qwen_factory=lambda run: _SpecQwen(chair))
    coordinator._run_d0(run)
    assert run.stage("D7").state == "skipped" and run.stage("D7").applicable is False

    # The human rolls back to D0 and Qwen now compiles an animated creature.
    StudioStore._invalidate_from(run, 0, "Reopened by a rollback from D1.")
    assert run.stage("D7").state == "skipped", "a contract skip survives invalidation"
    coordinator._qwen_factory = lambda run: _SpecQwen(footman_spec())
    coordinator._run_d0(run)

    for stage_id in ("D4", "D5", "D6", "D7"):
        stage = run.stage(stage_id)
        assert stage.applicable is True, f"{stage_id} kept the previous contract's exclusion"
        assert stage.state == "pending", f"{stage_id} stayed skipped under a spec that needs it"


def test_stop_flag_is_retired_once_the_stopped_job_ends(tmp_path: Path) -> None:
    """Regression: _stop_requested was only cleared by the next submit(), so
    after a stop finished the console kept reporting "waiting for the active
    worker to reach a safe stop point" beside an idle Resume button."""
    store = StudioStore(tmp_path)
    _awaiting_d1_run(store, "stop-clears")
    comfy = BlockingControlFakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit_manual_qwen_image(
            "stop-clears",
            "Render a natural original footman with an opaque shield and clearly gripped sword.",
        )
        assert comfy.started.wait(timeout=2)
        assert coordinator.stop("stop-clears")[0] is True
        assert coordinator.stopping("stop-clears") is True
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and coordinator.busy("stop-clears"):
            time.sleep(0.02)
        assert coordinator.busy("stop-clears") is False
        assert coordinator.stopping("stop-clears") is False, "the stop request outlived its job"


def test_direct_render_at_a_closed_gate_is_refused_not_failed(tmp_path: Path) -> None:
    """Regression: the gate precondition was only checked inside the worker
    thread, where its failure ran the generic handler and marked the D1 gate
    "failed" -- destroying a gate that was legitimately waiting for a human."""
    store = StudioStore(tmp_path)
    _awaiting_d1_run(store, "closed-gate")
    run = store.load("closed-gate")
    run.stage("D1").state = "approved"
    run.current_stage = "D2"
    store.save(run)
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(store, qwen_factory=lambda run: FakeQwen(), executor=executor)
        with pytest.raises(ValueError, match="awaits review"):
            coordinator.submit_manual_qwen_image("closed-gate", "Render a wholly different footman please.")
    unchanged = store.load("closed-gate")
    assert unchanged.stage("D1").state == "approved"
    assert unchanged.state != "failed"


def test_run_root_rejects_a_relative_path_segment(tmp_path: Path) -> None:
    """Regression: run_root() accepted any string of [a-z0-9._-], so ".."
    resolved to the studio root and /artifact/%2e%2e/<path> read files from
    outside the run directory."""
    store = StudioStore(tmp_path)
    for bad in ("..", ".", "...", "-nope"):
        with pytest.raises(ValueError, match="invalid run id"):
            store.run_root(bad)
    with pytest.raises(ValueError, match="invalid run id"):
        store.artifact_path("..", "runs")


def test_recovery_does_not_re_report_an_already_recovered_run(tmp_path: Path) -> None:
    """Regression: the recovery message itself matched the phrase recovery
    scanned for, so a failed-and-recovered run was 'recovered' again, and
    given another stage_recovered event, on every Studio launch."""
    store = StudioStore(tmp_path)
    run = store.create("interrupted-v1", DESCRIPTION)
    run.state = "running"
    run.stage("D2").state = "running"
    run.current_stage = "D2"
    store.save(run)

    assert store.recover_interrupted_runs() == ["interrupted-v1"]
    events_after_first = len(store.read_events("interrupted-v1"))
    assert store.recover_interrupted_runs() == []
    assert len(store.read_events("interrupted-v1")) == events_after_first


def test_d4_retry_overrides_reach_the_blender_rig_worker(tmp_path: Path) -> None:
    """Regression: D4's non-rigid path always sent landmark_adjustments={}
    and weight_adjustments=[] to the Blender worker, and render_size /
    maximum_material_change_fraction / maximum_bone_influences were read only
    from settings -- a human correction at the D4 gate had no way to reach
    the worker that actually proposes the rig."""
    store = StudioStore(tmp_path)
    worker = FakeWorkerExecutor()
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: FakeComfy(),
            worker_executor=worker,
            executor=executor,
        )
        store.create("d4-override-v1", DESCRIPTION)
        coordinator.submit("d4-override-v1")
        _approve_gate(store, coordinator, "d4-override-v1", "D1")
        run = wait_for(store, "d4-override-v1", "awaiting_review")
        assert run.current_stage == "D4"
        candidate = next(
            item for item in run.stage("D4").evidence if item.metrics.get("selectable") is True
        )
        store.decide(
            "d4-override-v1",
            "D4",
            "edit",
            "The left shoulder landmark sits too low; nudge it up and rebalance the weight split.",
            candidate.evidence_id,
            overrides={
                "landmark_adjustments": {"left_shoulder": [0.0, 0.0, 0.02]},
                "weight_adjustments": [
                    {
                        "joint_pair": "shoulder_left",
                        "direction": "parent_to_child",
                        "transfer_fraction": 0.1,
                        "radius_fraction": 0.1,
                    }
                ],
                "render_size": 640,
            },
        )
        coordinator.submit("d4-override-v1")
        wait_for(store, "d4-override-v1", "awaiting_review")

    rig_requests = [
        item for item in worker.requests if item.operation_id == "blender.propose_short_biped_rig"
    ]
    assert len(rig_requests) == 2, "the edit must trigger exactly one more D4 worker attempt"
    second = rig_requests[-1]
    assert second.parameters["landmark_adjustments"] == {"left_shoulder": [0.0, 0.0, 0.02]}
    assert second.parameters["weight_adjustments"] == [
        {
            "joint_pair": "shoulder_left",
            "direction": "parent_to_child",
            "transfer_fraction": 0.1,
            "radius_fraction": 0.1,
        }
    ]
    assert second.parameters["render_size"] == 640
    # consumed exactly once
    assert store.load("d4-override-v1").stage("D4").pending_overrides == {}
    # and an ordinary (non-override) attempt is unaffected -- the first
    # request must not have carried anything from the future correction
    first = rig_requests[0]
    assert first.parameters["landmark_adjustments"] == {}
    assert first.parameters["weight_adjustments"] == []


def test_d4_landmark_adjustment_shape_is_validated_at_the_gate(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run, item = _awaiting_stage_with_one_candidate(store, "d4-bad-override-v1", "D4")
    with pytest.raises(ValueError, match="landmark_adjustments"):
        store.decide(
            run.run_id,
            "D4",
            "edit",
            "fix it",
            item.evidence_id,
            overrides={"landmark_adjustments": {"left_shoulder": [0.0, 0.02]}},
        )
    with pytest.raises(ValueError, match="weight_adjustments"):
        store.decide(
            run.run_id,
            "D4",
            "edit",
            "fix it",
            item.evidence_id,
            overrides={"weight_adjustments": [1, 2, 3]},
        )
    with pytest.raises(ValueError, match="render_size"):
        store.decide(
            run.run_id, "D4", "edit", "fix it", item.evidence_id, overrides={"render_size": 8}
        )


def test_archiving_a_run_is_reversible_and_does_not_touch_state(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("archive-v1", DESCRIPTION)
    assert run.archived is False

    archived = store.set_archived("archive-v1", True)
    assert archived.archived is True
    assert archived.state == "created"  # unrelated to the state machine
    assert "run_archived" in [item["event_type"] for item in store.read_events("archive-v1")]

    unarchived = store.set_archived("archive-v1", False)
    assert unarchived.archived is False
    assert "run_unarchived" in [item["event_type"] for item in store.read_events("archive-v1")]

    # setting the same value again is a no-op, not a duplicate event
    before = len(store.read_events("archive-v1"))
    store.set_archived("archive-v1", False)
    assert len(store.read_events("archive-v1")) == before


def test_archived_runs_are_still_returned_by_list_and_still_recoverable(tmp_path: Path) -> None:
    """Archiving must only affect what the dashboard/CLI list chooses to show
    -- StudioStore.list() itself, and anything built on it like
    recover_interrupted_runs(), must keep seeing every run regardless."""
    store = StudioStore(tmp_path)
    store.create("archive-v2", DESCRIPTION)
    store.set_archived("archive-v2", True)
    assert "archive-v2" in [item.run_id for item in store.list()]

    run = store.load("archive-v2")
    run.state = "running"
    run.stage("D2").state = "running"
    run.current_stage = "D2"
    store.save(run)
    assert store.recover_interrupted_runs() == ["archive-v2"], "archiving must not hide a run from recovery"


def test_an_escalated_automatic_gate_can_actually_be_approved(tmp_path: Path) -> None:
    """Regression: when D2 exhausted its automatic retry budget it set
    gate_required and awaiting_review, and pointed its review at the latest
    diagnostic -- but never marked any evidence selectable. StudioStore.decide()
    refuses to approve unselectable evidence AND refuses to fall back to the
    recommendation for the same reason, so the human was handed a gate they
    could only reject or roll back. Escalating to a human must always leave
    something the human can say yes to."""
    store = StudioStore(tmp_path)
    run = store.create("escalated-v1", DESCRIPTION)
    stage = run.stage("D2")
    stage.gate_required = True
    stage.state = "awaiting_review"
    stage.iteration = 3
    diagnostic = store.run_root(run.run_id) / "D2_geometry" / "diagnostic.png"
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "grey").save(diagnostic)
    item = store.evidence(
        run,
        "D2",
        diagnostic,
        evidence_id="d2-i03-diagnostic",
        label="Geometry diagnostic",
        media_type="image/png",
        # exactly what _run_d2 writes when it escalates
        metrics={"iteration": 3, "selectable": True},
    )
    store.save(run)

    decided = store.decide(run.run_id, "D2", "approve", "Critic was too strict.", item.evidence_id)
    assert decided.stage("D2").state == "approved"
    assert decided.stage("D2").human_decisions[-1].selected_evidence_id == "d2-i03-diagnostic"


def test_unselectable_evidence_still_cannot_be_approved(tmp_path: Path) -> None:
    """The complement: the selectable flag must keep meaning something, so a
    report or comparison board is still refused."""
    store = StudioStore(tmp_path)
    run = store.create("escalated-v2", DESCRIPTION)
    stage = run.stage("D2")
    stage.gate_required = True
    stage.state = "awaiting_review"
    report = store.run_root(run.run_id) / "D2_geometry" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")
    item = store.evidence(
        run,
        "D2",
        report,
        evidence_id="d2-i03-report",
        label="Report",
        media_type="application/json",
        metrics={"iteration": 3, "selectable": False},
    )
    store.save(run)
    with pytest.raises(ValueError, match="production candidate"):
        store.decide(run.run_id, "D2", "approve", "", item.evidence_id)


def test_vram_handoff_classifies_an_absent_service_apart_from_a_refusing_one() -> None:
    """The handoff guarantees the *other* service is not holding VRAM. A
    refused TCP connection means that service is not running, so it holds
    none and the postcondition already holds. A service that answered and
    refused, or one that hung, must still fail closed.

    Regression for a real D0 failure: with ComfyUI not installed, every
    reviewer call died on `URLError: <urlopen error [WinError 10061] ...>`
    before it ever reached the LLM.
    """
    import socket
    import urllib.error

    refused = urllib.error.URLError(ConnectionRefusedError(10061, "actively refused it"))
    assert _VramHandoff._service_is_absent(refused) is True

    answered_and_refused = urllib.error.HTTPError(
        "http://127.0.0.1:8188/free", 500, "Internal Server Error", {}, None
    )
    assert _VramHandoff._service_is_absent(answered_and_refused) is False

    hung = urllib.error.URLError(socket.timeout("timed out"))
    assert _VramHandoff._service_is_absent(hung) is False


def test_vram_handoff_proceeds_when_the_other_service_is_not_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the proxy: an absent ComfyUI must not fail a
    reviewer call under a fail-closed device policy, while a ComfyUI that
    answers and refuses to unload still must."""
    import urllib.error

    from text2model_forge.studio_models import StudioRun

    monkeypatch.setattr("text2model_forge.studio_pipeline.wait_for_free_vram", lambda *a, **k: None)
    monkeypatch.setattr("text2model_forge.studio_pipeline.admit_gpu_memory", lambda *a, **k: None)
    monkeypatch.setattr("text2model_forge.studio_pipeline.gpu_memory_snapshot", lambda: None)
    # This test is about the unload of the *other* service, so hold the
    # service this call itself needs reachable; its own absence is covered
    # by test_a_heavyweight_call_reports_an_absent_service_before_waiting.
    monkeypatch.setattr(_VramHandoff, "_reachable", staticmethod(lambda url, timeout=2.0: True))

    run = StudioRun(
        run_id="handoff-probe",
        description=DESCRIPTION,
        comfy_url="http://127.0.0.1:9",
        localdeploy_url="http://127.0.0.1:9/v1",
        model="qwen3-vl:4b-instruct",
        vram_handoff=True,
        device_policy="gpu_compute_only",
        stages=[],
    )

    class Inner:
        def compile_spec(self, description: str) -> str:
            return "reached the reviewer"

    def absent(_url: str) -> None:
        raise urllib.error.URLError(ConnectionRefusedError(10061, "actively refused it"))

    monkeypatch.setattr(_VramHandoff, "_free_comfy", staticmethod(absent))
    proxy = _VramHandoff.wrap_qwen(Inner(), run, tmp_path)
    assert proxy.compile_spec(run.description) == "reached the reviewer"

    def refuses(_url: str) -> None:
        raise urllib.error.HTTPError("http://127.0.0.1:8188/free", 500, "boom", {}, None)

    monkeypatch.setattr(_VramHandoff, "_free_comfy", staticmethod(refuses))
    proxy = _VramHandoff.wrap_qwen(Inner(), run, tmp_path)
    with pytest.raises(urllib.error.HTTPError):
        proxy.compile_spec(run.description)


def test_a_heavyweight_call_reports_an_absent_service_before_waiting_on_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Order of checks. With ComfyUI not installed, D1 used to acquire the
    GPU lease, unload the reviewer, then wait out the full unload timeout
    and fail with "GPU memory did not recover within 60s" -- a message about
    a symptom, naming neither the missing service nor anything actionable.
    The reachability check must come first, and must not wait."""
    from text2model_forge.studio_models import StudioRun

    def must_not_run(*args, **kwargs):
        raise AssertionError("memory was waited on before the service was checked")

    monkeypatch.setattr("text2model_forge.studio_pipeline.wait_for_free_vram", must_not_run)
    monkeypatch.setattr("text2model_forge.studio_pipeline.admit_gpu_memory", must_not_run)

    run = StudioRun(
        run_id="absent-comfy",
        description=DESCRIPTION,
        comfy_url="http://127.0.0.1:9",
        localdeploy_url="http://127.0.0.1:9/v1",
        vram_handoff=True,
        device_policy="gpu_compute_only",
        stages=[],
    )

    class Inner:
        def generate(self, *, workflow, destination, timeout_seconds=900):
            raise AssertionError("ComfyUI was called even though it is not running")

    proxy = _VramHandoff.wrap_comfy(Inner(), run, tmp_path)
    with pytest.raises(RuntimeError, match="ComfyUI is not reachable"):
        proxy.generate(workflow={}, destination=tmp_path, timeout_seconds=5)


def test_concept_backend_envelopes_match_the_admission_gate() -> None:
    """preflight duplicates the coordinator's VRAM envelopes so it need not
    import it. If the two drift, preflight blesses a machine the admission
    gate will then refuse -- exactly the class of failure the check exists
    to prevent."""
    from text2model_forge.preflight import CONCEPT_BACKEND_VRAM_GB

    for backend, required in CONCEPT_BACKEND_VRAM_GB.items():
        assert _VramHandoff._COMFY_REQUIRED_GB[backend] == required, backend


def test_preflight_fails_when_no_concept_backend_can_fit_the_free_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The missing check behind every "it just keeps failing" report: an 8 GB
    laptop card whose desktop already holds ~2.6 GB can never reach the
    6.55 GB the cheapest image backend needs, so every run dies at D1. The
    reviewer's *currently resident* size counts as reclaimable; its nominal
    size must not, or an idle machine is reported as ready when it is not.
    """
    from text2model_forge.hardware import GpuInfo, HardwareProfile
    from text2model_forge.preflight import check_concept_backend_fits

    hardware = HardwareProfile(
        detected=True,
        source="nvidia-smi",
        gpus=[GpuInfo(name="RTX 3080 Laptop", backend="CUDA", vram_total_gb=8.0)],
    )
    settings = {
        "concept_backend": "sdxl",
        "gpu_safety_margin_gb": 0.75,
        "model": "qwen3-vl:4b-instruct",
        "localdeploy_url": "http://127.0.0.1:9/v1",
    }

    monkeypatch.setattr(
        "text2model_forge.preflight.gpu_memory_snapshot",
        lambda: {"device": {"total_gb": 8.0, "free_gb": 5.22}},
    )
    monkeypatch.setattr("text2model_forge.preflight._http_json", lambda url, timeout=8.0: None)

    failing = check_concept_backend_fits(hardware, settings)
    assert failing.status == "fail"
    assert "6.55" in failing.detail and "5.22" in failing.detail
    assert failing.remedy

    # Same machine, but the reviewer is resident and will be unloaded first.
    monkeypatch.setattr(
        "text2model_forge.preflight._http_json",
        lambda url, timeout=8.0: {"models": [{"size_vram": 2.0 * 1024**3}]},
    )
    with_reclaim = check_concept_backend_fits(hardware, settings)
    assert with_reclaim.status == "ok"


def test_progress_telemetry_path_executes_without_a_missing_import(tmp_path: Path) -> None:
    """A real D1 run crashed with `NameError: name 'time' is not defined`
    inside _progress()'s GPU-telemetry throttle. The whole deterministic
    suite passed anyway, because every fake-backed test leaves telemetry
    disabled, so the two lines that reference `time` never executed. This
    drives _progress() with telemetry ON, which is the only way that branch
    is reached.
    """
    store = StudioStore(tmp_path)
    run = store.create("telemetry-probe", DESCRIPTION, {})
    coordinator = StudioCoordinator(
        store,
        qwen_factory=lambda run: FakeQwen(),
        comfy_factory=lambda run: FakeComfy(),
        worker_executor=FakeWorkerExecutor(),
    )
    try:
        coordinator._telemetry_enabled = True
        coordinator._last_gpu_snapshot_at = 0.0
        coordinator._progress(run, "D0", 0.5, "probing the telemetry branch")
    finally:
        coordinator.close()

    assert run.stage("D0").message == "probing the telemetry branch"


def test_failed_atomic_save_restores_the_in_memory_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StudioStore(tmp_path)
    run = store.create("failed-save", DESCRIPTION, {})
    original_revision = run.revision
    original_updated_at = run.updated_at
    original_write_text = Path.write_text

    def fail_temporary_write(path: Path, *args, **kwargs):
        if path.name == "run.json.tmp":
            raise OSError("simulated disk failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_temporary_write)
    run.title = "This mutation must remain unsaved"

    with pytest.raises(OSError, match="simulated disk failure"):
        store.save(run)

    assert run.revision == original_revision
    assert run.updated_at == original_updated_at
    assert store.load(run.run_id).title != run.title


def test_failed_event_save_removes_the_uncommitted_jsonl_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StudioStore(tmp_path)
    run = store.create("failed-event", DESCRIPTION, {})
    events_path = store.run_root(run.run_id) / "events.jsonl"
    original_events = events_path.read_bytes()
    original_count = run.event_count

    def fail_save(_run):
        raise OSError("simulated run save failure")

    monkeypatch.setattr(store, "save", fail_save)
    with pytest.raises(OSError, match="simulated run save failure"):
        store.event(run, "uncommitted_event", {"value": 1})

    assert run.event_count == original_count
    assert events_path.read_bytes() == original_events


def test_gpu_only_policy_rejects_a_comfyui_that_streams_weights_from_ram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode with no error message. A ComfyUI started with
    --lowvram renders correctly but streams weights across PCIe: measured
    at 8m52s of model initialization for 30s of sampling, against 45s for
    the same render resident. Nothing logged a problem, so it read as "the
    tool is broken". gpu_compute_only already forbade CPU *nodes*; it now
    forbids a CPU-offloading *server* too, which is the same policy applied
    to the thing that actually decides where the weights live.
    """
    from text2model_forge.studio_models import StudioRun

    run = StudioRun(
        run_id="offload-guard",
        description=DESCRIPTION,
        comfy_url="http://127.0.0.1:8188",
        localdeploy_url="http://127.0.0.1:11434/v1",
        vram_handoff=True,
        device_policy="gpu_compute_only",
        stages=[],
    )

    monkeypatch.setattr(
        _VramHandoff, "_comfy_offload_flags", staticmethod(lambda url: ["--lowvram"])
    )

    class Inner:
        def generate(self, *, workflow, destination, timeout_seconds=900):
            raise AssertionError("a streaming ComfyUI must not be used under a GPU-only policy")

    proxy = _VramHandoff.wrap_comfy(Inner(), run, tmp_path)
    with pytest.raises(RuntimeError, match="--lowvram"):
        proxy.generate(workflow={"1": {"class_type": "KSampler", "inputs": {}}}, destination=tmp_path)

    # prefer_gpu is the explicitly degraded mode and must still allow it.
    monkeypatch.setattr(_VramHandoff, "_reachable", staticmethod(lambda url, timeout=2.0: True))
    monkeypatch.setattr(_VramHandoff, "_comfy_free_gb", staticmethod(lambda url: 99.0))
    lenient = run.model_copy(update={"device_policy": "prefer_gpu"})

    class Ok:
        def generate(self, *, workflow, destination, timeout_seconds=900):
            return ["rendered"]

    assert _VramHandoff.wrap_comfy(Ok(), lenient, tmp_path).generate(
        workflow={"1": {"class_type": "KSampler", "inputs": {}}}, destination=tmp_path
    ) == ["rendered"]


def test_preflight_reports_a_streaming_comfyui_before_the_run_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same guard, one step earlier: caught in the readiness report
    rather than minutes into a render."""
    from text2model_forge.preflight import check_comfy_memory_mode

    monkeypatch.setattr(
        "text2model_forge.preflight._http_json",
        lambda url, timeout=8.0: {"system": {"argv": ["main.py", "--lowvram", "--port", "8188"]}},
    )
    strict = check_comfy_memory_mode(
        {"comfy_url": "http://127.0.0.1:8188", "device_policy": "gpu_compute_only"}
    )
    assert strict.status == "fail"
    assert "--lowvram" in strict.detail and strict.remedy

    lenient = check_comfy_memory_mode(
        {"comfy_url": "http://127.0.0.1:8188", "device_policy": "prefer_gpu"}
    )
    assert lenient.status == "warn"

    monkeypatch.setattr(
        "text2model_forge.preflight._http_json",
        lambda url, timeout=8.0: {"system": {"argv": ["main.py", "--reserve-vram", "0.4"]}},
    )
    assert check_comfy_memory_mode({"comfy_url": "http://127.0.0.1:8188"}).status == "ok"
