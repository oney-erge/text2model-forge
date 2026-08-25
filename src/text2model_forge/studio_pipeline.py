"""Asynchronous, resumable execution of Text2Model Studio stages."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from typing import Any, Protocol
import urllib.error
import urllib.request
from text2model_forge.paths import resource_root

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat

from .config import load_local_config, worker_binding
from .concept_quality import assess_concept_image
from .duplicate_detection import patch_already_present_elsewhere
from .external_worker import SubprocessWorkerAdapter
from .gpu import (
    GpuLease,
    GpuMemoryAdmissionError,
    admit_gpu_memory,
    gpu_memory_snapshot,
    wait_for_free_vram,
)
from .hardware import reviewer_vram_requirement
from .hashing import sha256_file
from .manifests import load_manifests
from .schemas import ArtifactLineage, ArtifactRecord, AssetStage, ExternalWorkerRequest
from .spec_conformance import (
    EquipmentConformanceReport,
    check_equipment_conformance,
    presence_check,
)
from .studio_comfy import (
    StudioComfyError,
    StudioComfyClient,
    concept_workflow,
    explicit_cpu_nodes,
    hunyuan3d_workflow,
    inpaint_workflow,
    make_chroma_alpha,
    make_humanoid_openpose_guide,
    qwen_image_2512_workflow,
    z_image_turbo_workflow,
)
from .motion_library import load_motion_library, resolve_donor_motion_path
from .settings import resolve_settings
from .studio_models import (
    StudioQwenReview,
    StudioRun,
    awaiting_correction,
    latest_correction,
    utc_now,
    validate_stage_overrides,
)
from .studio_qwen import ConceptCorrectionPlan, ConceptPlan, StudioQwen
from .studio_store import StudioStore
from .workers import WorkerManager


class QwenProvider(Protocol):
    def compile_spec(self, description: str): ...
    def concept_plan(self, spec, stage): ...
    def concept_correction_plan(self, spec, stage, candidate_ids, comparison_board=None): ...
    def review_concepts(self, spec, stage, images, comparison_board=None): ...
    def revision_plan(self, spec, stage): ...


class ComfyProvider(Protocol):
    def checkpoints(self) -> list[str]: ...
    def controlnets(self) -> list[str]: ...
    def models(self, kind: str) -> list[str]: ...
    def upload_image(self, name: str, data: bytes, subfolder: str = "text2model_studio") -> str: ...
    def generate(self, *, workflow, destination: Path, timeout_seconds: float = 900) -> list[Path]: ...


def _motion_catalog_path() -> Path | None:
    """The motion catalog, preferring a project's own over the packaged one --
    the same working-directory-first rule profiles/ uses."""
    for candidate in (
        Path.cwd() / "motion_library" / "catalog.json",
        resource_root() / "motion_library" / "catalog.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _image_metrics(path: Path) -> dict[str, float | int | bool | str | None]:
    with Image.open(path).convert("RGB") as image:
        stats = ImageStat.Stat(image)
        return {
            "width": image.width,
            "height": image.height,
            "aspect_ratio": round(image.width / image.height, 4),
            "mean_luminance_255": round(
                0.2126 * stats.mean[0] + 0.7152 * stats.mean[1] + 0.0722 * stats.mean[2], 3
            ),
            "nonblank": max(stats.var) > 1.0,
        }


def _concept_comparison_board(
    store: StudioStore,
    run: StudioRun,
    stage,
    candidates: list[tuple[str, Path, dict[str, object]]],
    output: Path,
) -> Path:
    records: list[tuple[str, Path]] = []
    latest_rejection = latest_correction(stage)
    if latest_rejection and latest_rejection.selected_evidence_id:
        prior = next(
            (
                item
                for item in stage.evidence
                if item.evidence_id == latest_rejection.selected_evidence_id
            ),
            None,
        )
        if prior is not None and prior.media_type.startswith("image/"):
            records.append((f"PREVIOUS REJECTED: {prior.evidence_id}", store.artifact_path(run.run_id, prior.relative_path)))
    records.extend((f"CURRENT: {evidence_id}", path) for evidence_id, path, _ in candidates)
    cell_width, cell_height, detail_height, header = 480, 640, 230, 62
    board = Image.new(
        "RGB",
        (cell_width * len(records), cell_height + detail_height + header + 34),
        "#10161a",
    )
    draw = ImageDraw.Draw(board)
    for index, (label, path) in enumerate(records):
        with Image.open(path).convert("RGB") as image:
            fitted = ImageOps.contain(image, (cell_width - 16, cell_height - 16))
            crop = image.crop(
                (0, round(image.height * 0.27), image.width, round(image.height * 0.72))
            )
            detail = ImageOps.contain(crop, (cell_width - 16, detail_height - 25))
        x = index * cell_width + (cell_width - fitted.width) // 2
        y = header + (cell_height - fitted.height) // 2
        board.paste(fitted, (x, y))
        detail_x = index * cell_width + (cell_width - detail.width) // 2
        detail_y = header + cell_height + 27
        board.paste(detail, (detail_x, detail_y))
        draw.text(
            (index * cell_width + 10, header + cell_height + 7),
            "HAND / EQUIPMENT DETAIL",
            fill="#9ed8ed",
        )
        draw.text((index * cell_width + 10, 15), label, fill="white")
        if index == 0 and latest_rejection:
            draw.text(
                (index * cell_width + 10, 36),
                ("HUMAN: " + latest_rejection.comment)[:72],
                fill="#ffb59f",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output)
    return output


def _prepare_inpaint_crop(
    source: Path,
    box: list[float],
    output_directory: Path,
) -> tuple[Path, Path, tuple[int, int, int, int]]:
    """Expand a defect box, scale it for detail work, and mask only the requested defect."""
    with Image.open(source).convert("RGB") as image:
        width, height = image.size
    x0, y0, x1, y1 = box
    defect = (
        max(0, min(width - 1, round(x0 * width))),
        max(0, min(height - 1, round(y0 * height))),
        max(1, min(width, round(x1 * width))),
        max(1, min(height, round(y1 * height))),
    )
    defect = (defect[0], defect[1], max(defect[0] + 1, defect[2]), max(defect[1] + 1, defect[3]))
    margin_x = max(24, round((defect[2] - defect[0]) * 0.16))
    margin_y = max(24, round((defect[3] - defect[1]) * 0.12))
    crop_box = (
        max(0, defect[0] - margin_x),
        max(0, defect[1] - margin_y),
        min(width, defect[2] + margin_x),
        min(height, defect[3] + margin_y),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    target_size = (768, 1024)
    with Image.open(source).convert("RGB") as image:
        crop = image.crop(crop_box).resize(target_size, Image.Resampling.LANCZOS)
    crop_path = output_directory / "correction_crop.png"
    crop.save(crop_path)
    scale_x = target_size[0] / (crop_box[2] - crop_box[0])
    scale_y = target_size[1] / (crop_box[3] - crop_box[1])
    mask_box = (
        round((defect[0] - crop_box[0]) * scale_x),
        round((defect[1] - crop_box[1]) * scale_y),
        round((defect[2] - crop_box[0]) * scale_x),
        round((defect[3] - crop_box[1]) * scale_y),
    )
    mask = Image.new("L", target_size, 0)
    ImageDraw.Draw(mask).rectangle(mask_box, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
    mask_path = output_directory / "correction_mask.png"
    mask.save(mask_path)
    return crop_path, mask_path, crop_box


def _composite_inpaint_crop(
    source: Path,
    generated_crop: Path,
    mask: Path,
    crop_box: tuple[int, int, int, int],
    output: Path,
) -> Path:
    crop_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    with Image.open(source).convert("RGB") as base:
        with Image.open(generated_crop).convert("RGB") as generated:
            repaired = generated.resize(crop_size, Image.Resampling.LANCZOS)
        with Image.open(mask).convert("L") as mask_image:
            feather = mask_image.resize(crop_size, Image.Resampling.LANCZOS)
        original_crop = base.crop(crop_box)
        merged = Image.composite(repaired, original_crop, feather)
        base.paste(merged, crop_box[:2])
        output.parent.mkdir(parents=True, exist_ok=True)
        base.save(output)
    return output


def _deferred_shield_equipment(spec):
    """A shield is deferred to a dedicated local-repair pass rather than described in the global
    prompt. A rectangular held prop (shield) fits a box-shaped inpaint mask naturally; a thin
    diagonal prop (sword) does not and degrades into a flat rectangular panel if forced through
    the same mechanism, so the global pass renders it directly instead. Mentioning both props in
    one prompt also measurably increases prop-bleed (mixed/duplicated weapons) in testing, so the
    global pass is told the shield arm is empty rather than asked to render two held props."""
    return next((item for item in spec.equipment if item.category == "shield"), None)


def _limb_repair_box(side: str) -> list[float]:
    """A tight box around the forearm/hand only (not a full body-half); tuned empirically so the
    inpaint has enough room for a shield without also including so much open background that the
    model reinterprets the whole masked region as a flat surface."""
    return [0.50, 0.30, 0.94, 0.72] if side == "left" else [0.06, 0.30, 0.50, 0.72]


def _deferred_shield_repair_prompts(item) -> tuple[str, str]:
    positive = (
        "close-up production concept of the existing character's armored arm and hand, "
        f"(one complete flat round shield, disc-shaped, strapped to the forearm and gripped by the hand, "
        f"{item.description}:1.5), matching the existing armor color and lighting, sharp focus"
    )
    negative = (
        "grainy, blurry, low quality, empty hand, no shield, sword, weapon, extra fingers, deformed hand, "
        "different armor color, different style, two shields, barrel, keg, cask, cylinder, crate, drum, "
        "wine barrel, wooden box"
    )
    return positive, negative


def _whole_asset_prompt_guard(
    spec, positive: str, negative: str, *, defer_shield: bool = True
) -> tuple[str, str]:
    """Keep a correction prompt from accidentally turning into an isolated prop sheet.

    Handedness is stated once in plain language and reinforced by the OpenPose skeleton at
    generation time; there is no region-compositing here, so there is only ever one body. Any
    For the ordinary SDXL route the shield may be deliberately left out (see
    _deferred_shield_equipment) and added by a local-repair pass. Qwen Image Edit receives
    the complete composition guide and must keep both pieces together, so it disables that
    SDXL-only deferral.
    """
    if spec.asset_kind != "character":
        return positive, negative
    materials = ", ".join(spec.materials[:4])
    deferred_shield = _deferred_shield_equipment(spec) if defer_shield else None
    deferred_ids = {deferred_shield.equipment_id} if deferred_shield else set()
    right_equipment = [
        item for item in spec.equipment if item.side == "right" and item.equipment_id not in deferred_ids
    ]
    left_equipment = [
        item for item in spec.equipment if item.side == "left" and item.equipment_id not in deferred_ids
    ]
    right_text = "; ".join(item.description for item in right_equipment)
    left_text = "; ".join(item.description for item in left_equipment)
    left_has_forearm_shield = any(item.category == "shield" for item in left_equipment)
    guarded_positive = (
        "ONE SINGLE COMPLETE LIVING FULL-BODY HUMAN CHARACTER, not an equipment display: a stylized chunky heroic "
        "figure standing centered in a neutral A-pose, with one connected head, face, torso, two shoulders, two "
        "attached arms, two attached hands, two legs, and boots all visible inside frame. Three-quarter-front "
        f"production concept on a plain studio background. The same person wears {materials}. "
        + (
            f"(the character's right hand firmly gripping the hilt of {right_text}, clearly extended away from "
            "the body and away from the shield-side arm:1.4). "
            if right_text
            else ""
        )
        + (
            f"(the character's left forearm visibly and securely strapped into {left_text}, with the left hand "
            "shown at the shield straps and not holding a second weapon:1.4). "
            if left_has_forearm_shield
            else f"(the character's left hand firmly gripping the hilt of {left_text}, clearly extended away from "
            "the body:1.4). "
            if left_text
            else ""
        )
        + (
            "The character's other arm and hand are relaxed and empty at the side, without any shield or extra "
            "prop. "
            if deferred_shield
            else ""
        )
        + positive
    )
    guarded_negative = (
        "isolated equipment, equipment inventory, weapon display, armor rack, wall-mounted swords, disembodied "
        "armor, empty suit of armor, missing human body, invisible person, prop sheet, multiple characters, "
        "extra limbs, extra arms, duplicated body, floating weapon, floating shield, "
        + ("empty right hand, unarmed, weapon on ground, weapon on back, sheathed weapon, " if right_text else "")
        + ("empty left hand, unarmed, " if left_text else "")
        + ("shield, " if deferred_shield else "")
        + negative
    )
    return guarded_positive, guarded_negative


def _geometry_ready_backdrop_guard(positive: str, negative: str) -> tuple[str, str]:
    """Request a background that can be removed deterministically before D2.

    D1's hard gate measures the resulting alpha instead of trusting this text,
    but every image backend must receive the same explicit production contract.
    """
    return (
        positive
        + " Isolated against one flat, evenly lit chroma-key green (#00ff00) background, "
        "with a clean silhouette and no cast shadow.",
        "complex background, scenery, floor line, horizon, cast shadow, green clothing, "
        "green material, green reflections, "
        + negative,
    )


def _image_board(records: list[tuple[str, Path]], output: Path, *, columns: int = 4) -> Path:
    if not records:
        raise ValueError("an evidence board requires at least one image")
    cell_width, cell_height, header = 360, 420, 44
    columns = max(1, min(columns, len(records)))
    rows = (len(records) + columns - 1) // columns
    board = Image.new("RGB", (columns * cell_width, rows * (cell_height + header)), "#10161a")
    draw = ImageDraw.Draw(board)
    for index, (label, path) in enumerate(records):
        column, row = index % columns, index // columns
        with Image.open(path).convert("RGB") as source:
            image = ImageOps.contain(source, (cell_width - 16, cell_height - 16))
        x = column * cell_width + (cell_width - image.width) // 2
        y = row * (cell_height + header) + header + (cell_height - image.height) // 2
        board.paste(image, (x, y))
        draw.text((column * cell_width + 9, row * (cell_height + header) + 13), label, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output)
    return output


def _rigid_structure_overlay(source: Path, plan, output: Path) -> Path:
    with Image.open(source).convert("RGB") as image:
        canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = ("#ff9e72", "#64d4ff", "#d4e65d", "#d487ff")
    for index, part in enumerate(plan.parts):
        x0, y0, x1, y1 = part.front_box_normalized
        box = (x0 * canvas.width, y0 * canvas.height, x1 * canvas.width, y1 * canvas.height)
        color = colors[index % len(colors)]
        draw.rectangle(box, outline=color, width=max(2, canvas.width // 300))
        pivot = (part.pivot_normalized[0] * canvas.width, part.pivot_normalized[1] * canvas.height)
        radius = max(5, canvas.width // 80)
        draw.ellipse(
            (pivot[0] - radius, pivot[1] - radius, pivot[0] + radius, pivot[1] + radius),
            outline=color,
            width=max(2, canvas.width // 300),
        )
        draw.text(
            (box[0] + 4, box[1] + 4),
            f"{part.component_id} | {part.rotation_axis} | {part.minimum_degrees:g}..{part.maximum_degrees:g}",
            fill=color,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


class _VramHandoff:
    """One fail-closed lease around every heavyweight HTTP inference call.

    The Studio already assumes one GPU-heavy stage at a time, and that is
    enough on a 24 GB card. It is not enough on 8 GB, because D1 alternates
    *within* one stage: Qwen plans the prompts, ComfyUI renders, Qwen
    critiques the renders. Measured on an RTX 3080 Laptop (8 GB), an idle
    ComfyUI plus a resident 8B reviewer leaves 250 MiB free, ollama spills to
    CPU, and D1's planning call exceeds even a 600 s timeout. Freeing both
    services returns ~7 GB, so the capacity is there -- it just has to be
    handed over rather than shared.

    The old handoff swallowed unload errors and then launched anyway. That is
    precisely how a nominally GPU run spills to CPU or OOMs. In a GPU-only
    policy, unload, live-memory measurement, and workflow device validation
    are admission conditions. ``prefer_gpu`` retains an explicit degraded
    mode for machines without supported telemetry.
    """

    _COMFY_REQUIRED_GB = {
        "z_image_turbo": 6.5,
        "qwen_image_2512": 6.5,
        "qwen_image_edit_2511": 6.5,
        "sdxl": 5.8,
        "hunyuan3d": 6.0,
    }

    @staticmethod
    def _free_comfy(comfy_url: str) -> None:
        request = urllib.request.Request(
            comfy_url.rstrip("/") + "/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=20).close()

    @staticmethod
    def _free_llm(localdeploy_url: str, model: str) -> None:
        # Ollama's own endpoint, not the OpenAI-compatible one: keep_alive=0
        # is how a model is asked to unload immediately.
        root = localdeploy_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        request = urllib.request.Request(
            root + "/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=20).close()

    @staticmethod
    def _admission_advice(original: str, required_gb: float, margin_gb: float) -> str:
        """Turn a bare admission refusal into something the operator can act on.

        "GPU memory did not recover within 60s; last free VRAM was 0.601 GiB"
        is true and unusable: it names neither how much was needed nor what
        is holding the card. On a laptop GPU the answer is usually the
        desktop itself -- a browser, an editor, and a vendor overlay can
        hold 2-3 GB of an 8 GB card before any model loads, which is enough
        to make every image backend permanently unadmittable.
        """
        needed = required_gb + margin_gb
        detail = [original, f"This step needs {needed:.2f} GB free ({required_gb:.2f} GB + {margin_gb:.2f} GB margin)."]
        snapshot = None
        try:
            snapshot = gpu_memory_snapshot()
        except Exception:
            snapshot = None
        device = (snapshot or {}).get("device") or {}
        if device:
            total = float(device.get("total_gb") or 0)
            free = float(device.get("free_gb") or 0)
            if total:
                held = total - free
                detail.append(
                    f"The card has {total:.2f} GB total and {free:.2f} GB free right now, "
                    f"so {held:.2f} GB is held by other processes."
                )
                if needed > total:
                    detail.append(
                        "That is more than the card's entire capacity, so this backend cannot run "
                        "on this GPU at any desktop load. Choose a smaller concept backend."
                    )
                elif held > 0.5:
                    detail.append(
                        "Close GPU-using applications (a browser, an editor, or a vendor overlay are "
                        "the usual holders) and resume, or choose a smaller concept backend. "
                        "The System page lists what this machine can actually reach."
                    )
        return " ".join(detail)

    @staticmethod
    def _endpoint(run: StudioRun, service: str) -> tuple[str, str, str]:
        """(display name, health URL, remedy) for the service a call will use."""
        if service == "comfy":
            base = run.comfy_url.rstrip("/")
            return (
                "ComfyUI",
                f"{base}/system_stats",
                "Start ComfyUI: python main.py --listen 127.0.0.1 --port 8188 "
                "--normalvram --reserve-vram 0.75 --preview-method none",
            )
        base = run.localdeploy_url.rstrip("/")
        return ("The reviewer model", f"{base}/models", "Start Ollama: ollama serve")

    @staticmethod
    def _reachable(url: str, timeout: float = 2.0) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.status == 200
        except Exception:
            return False

    @staticmethod
    def _service_is_absent(exc: BaseException) -> bool:
        """True only when the unload call failed because nothing is listening.

        This is not a relaxation of the fail-closed admission contract, and
        the distinction is the whole point. The handoff exists to guarantee
        one postcondition -- the *other* service is not holding VRAM when
        this one loads. A service that refused the TCP connection is not
        running, therefore holds no VRAM, therefore already satisfies that
        postcondition. Failing the stage there reports a memory-safety
        violation that provably cannot exist, which is what turned "ComfyUI
        is not installed yet" into a D0 stage failure carrying a raw
        WinError 10061.

        Two cases are deliberately excluded and still fail closed:

        - an HTTPError means the service answered and *refused* to unload,
          which is a real admission failure;
        - a timeout means the service is listening but not responding, and
          a hung service may well still be resident and holding memory --
          exactly the condition this handoff protects against.

        The live-telemetry admission checks below (wait_for_free_vram and
        admit_gpu_memory) are untouched and still gate on measured free
        memory, so a machine that really is short on VRAM still fails here.
        """
        if isinstance(exc, urllib.error.HTTPError):
            return False
        reason = getattr(exc, "reason", exc)
        return isinstance(reason, ConnectionError)

    class _Proxy:
        def __init__(self, inner: Any, run: StudioRun, lease_root: Path, service: str) -> None:
            self._inner = inner
            self._run = run
            self._lease_root = lease_root
            self._service = service

        def _resident_gb(self) -> float:
            """How much VRAM the service this call targets already holds.

            Only measurable for the reviewer, whose server reports per-model
            residency. ComfyUI exposes no equivalent per-model figure, so its
            side returns 0 and keeps demanding the full envelope -- the
            conservative direction, and harmless because the handoff unloads
            ComfyUI before a reviewer call anyway.
            """
            if self._service != "qwen":
                return 0.0
            root = self._run.localdeploy_url.rstrip("/")
            if root.endswith("/v1"):
                root = root[: -len("/v1")]
            try:
                with urllib.request.urlopen(root + "/api/ps", timeout=4) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception:
                return 0.0
            resident = 0.0
            for item in (payload or {}).get("models") or []:
                if str(item.get("name")) != self._run.model:
                    continue
                size = item.get("size_vram")
                if isinstance(size, (int, float)):
                    resident += float(size) / (1024**3)
            return resident

        def __getattr__(self, name: str) -> Any:
            attribute = getattr(self._inner, name)
            if not callable(attribute):
                return attribute

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                heavyweight = self._service == "qwen" or name == "generate"
                if not heavyweight:
                    return attribute(*args, **kwargs)
                workflow = kwargs.get("workflow") if self._service == "comfy" else None
                if (
                    self._run.device_policy in {"gpu_compute_only", "strict_device_only"}
                    and isinstance(workflow, dict)
                ):
                    cpu_nodes = explicit_cpu_nodes(workflow)
                    if cpu_nodes:
                        raise RuntimeError(
                            "GPU-compute-only policy rejected a ComfyUI workflow with explicit CPU "
                            f"inference nodes: {cpu_nodes}"
                        )
                    # The same policy, applied to the server rather than the
                    # graph. A ComfyUI started with --lowvram/--novram/--cpu
                    # offloads weights to system RAM: nothing fails, it just
                    # streams over PCIe, which is the silent CPU fallback
                    # this policy forbids. Checked here so it can never be
                    # reintroduced by however ComfyUI happens to be launched.
                    offload = _VramHandoff._comfy_offload_flags(self._run.comfy_url)
                    if offload:
                        raise RuntimeError(
                            f"GPU-compute-only policy rejected ComfyUI: it is running with "
                            f"{' '.join(offload)}, which streams model weights from system RAM "
                            "instead of keeping them on the GPU. Restart it without those flags "
                            "(--reserve-vram is fine), or set [studio_defaults].device_policy = "
                            '"prefer_gpu" to accept the slowdown deliberately.'
                        )
                required = (
                    reviewer_vram_requirement(self._run.model) or 4.0
                    if self._service == "qwen"
                    else _VramHandoff._comfy_required(self._run, workflow)
                )
                # Confirm the service exists before spending up to a minute
                # waiting for GPU memory on its behalf. An absent ComfyUI
                # used to surface as "GPU memory did not recover within 60s",
                # which names the wrong problem entirely and hides the one
                # thing the operator can act on. A refused loopback
                # connection costs microseconds, so this is free when the
                # service is up.
                name_, health_url, remedy = _VramHandoff._endpoint(self._run, self._service)
                if not _VramHandoff._reachable(health_url):
                    raise RuntimeError(
                        f"{name_} is not reachable at {health_url}, so this stage cannot run. {remedy}"
                    )
                lease = GpuLease(
                    self._lease_root,
                    run_id=self._run.run_id,
                    worker_id=f"studio.{self._service}",
                )
                lease.acquire(timeout_seconds=max(30, self._run.llm_timeout_seconds))
                try:
                    if self._run.vram_handoff:
                        try:
                            if self._service == "qwen":
                                # Evict ComfyUI only when the reviewer will
                                # not otherwise fit. Unconditional eviction
                                # made D1 pathological: ComfyUI re-staged
                                # ~5.9 GB from disk before every subsequent
                                # render, measured at 8m52s of model
                                # initialization for 30s of actual sampling,
                                # on a stage that renders six candidates and
                                # alternates with the reviewer between each.
                                snapshot = gpu_memory_snapshot() or {}
                                free_now = float(
                                    (snapshot.get("device") or {}).get("free_gb") or 0
                                )
                                needed = required + self._run.gpu_safety_margin_gb
                                if free_now < needed:
                                    _VramHandoff._free_comfy(self._run.comfy_url)
                            else:
                                _VramHandoff._free_llm(self._run.localdeploy_url, self._run.model)
                        except Exception as exc:
                            if (
                                self._run.device_policy != "prefer_gpu"
                                and not _VramHandoff._service_is_absent(exc)
                            ):
                                raise
                    # Memory this service ALREADY holds is its own allocation,
                    # not competition for it. Without this the reviewer waits
                    # for room to load a model that is already loaded: with
                    # qwen3-vl:4b resident at 3.94 GB and 0.42 GB free, the
                    # gate demanded 4.35 GB free *for that same model* and
                    # timed out every time. Deducting what is already resident
                    # is not a weakening -- an unloaded model still has to
                    # prove its full envelope fits.
                    outstanding = max(0.0, required - self._resident_gb())
                    # ComfyUI's own --reserve-vram already holds back driver
                    # headroom. Studio's safety margin serves the same
                    # purpose, so demanding both reserved it twice and
                    # refused renders that fit with room to spare.
                    margin = self._run.gpu_safety_margin_gb
                    self_reported: float | None = None
                    if self._service == "comfy":
                        margin = max(
                            0.0, margin - _VramHandoff._comfy_reserved_gb(self._run.comfy_url)
                        )
                        self_reported = _VramHandoff._comfy_free_gb(self._run.comfy_url)
                    if self_reported is not None:
                        # ComfyUI is the process that will allocate, and it
                        # counts its own reusable pools. Take its answer
                        # rather than the driver's, which cannot see them.
                        if self_reported < outstanding + margin:
                            raise GpuMemoryAdmissionError(
                                _VramHandoff._admission_advice(
                                    f"ComfyUI reports only {self_reported:.2f} GiB allocatable",
                                    outstanding,
                                    margin,
                                )
                            )
                        return attribute(*args, **kwargs)
                    if self._run.vram_handoff and self._run.device_policy != "prefer_gpu":
                        if outstanding > 0:
                            try:
                                wait_for_free_vram(
                                    outstanding,
                                    safety_margin_gb=margin,
                                    timeout_seconds=self._run.gpu_unload_timeout_seconds,
                                )
                            except GpuMemoryAdmissionError as exc:
                                raise GpuMemoryAdmissionError(
                                    _VramHandoff._admission_advice(
                                        str(exc), outstanding, margin
                                    )
                                ) from exc
                    # Nothing to admit when the service already holds its
                    # whole envelope: there is no new allocation to prove
                    # fits. Demanding the driver-headroom margin anyway
                    # refused a resident reviewer over 0.02 GB -- the same
                    # model that had just completed D0 on that exact card.
                    # The margin protects a *load*; this call performs none.
                    if outstanding > 0:
                        try:
                            admit_gpu_memory(
                                outstanding,
                                safety_margin_gb=margin,
                                require_measurement=self._run.device_policy != "prefer_gpu",
                            )
                        except GpuMemoryAdmissionError as exc:
                            raise GpuMemoryAdmissionError(
                                _VramHandoff._admission_advice(
                                    str(exc), outstanding, margin
                                )
                            ) from exc
                    return attribute(*args, **kwargs)
                finally:
                    lease.release()

            return wrapped

    # How much of a model ComfyUI actually keeps resident, by the memory
    # mode it was launched with. The envelopes above describe the default
    # mode, where the whole model is loaded to the device. --lowvram and
    # --novram stream weights block by block, so peak residency is a
    # fraction of the file: refusing a job on the full-load figure while
    # ComfyUI is streaming rejects work the GPU can comfortably do.
    # Measured from ComfyUI's own reported argv rather than assumed, so a
    # relaunch in a different mode is picked up without touching config.
    _COMFY_MODE_FRACTION = {"--novram": 0.25, "--lowvram": 0.45, "--highvram": 1.0, "--gpu-only": 1.0}

    @staticmethod
    def _comfy_mode_fraction(comfy_url: str) -> float:
        try:
            with urllib.request.urlopen(
                comfy_url.rstrip("/") + "/system_stats", timeout=4
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return 1.0
        argv = [str(item) for item in (payload.get("system") or {}).get("argv") or []]
        for flag, fraction in _VramHandoff._COMFY_MODE_FRACTION.items():
            if flag in argv:
                return fraction
        return 1.0

    # ComfyUI launch flags that move weights off the GPU. Under a
    # GPU-only device policy these are exactly the "silent CPU inference"
    # the policy exists to forbid: the run still completes, so nothing
    # fails, it just streams weights across PCIe and takes forever --
    # measured here at 8m52s of model initialization for 30s of sampling,
    # against 45s for the same render with the model resident.
    _COMFY_CPU_OFFLOAD_FLAGS = ("--cpu", "--novram", "--lowvram")

    @staticmethod
    def _comfy_offload_flags(comfy_url: str) -> list[str]:
        try:
            with urllib.request.urlopen(
                comfy_url.rstrip("/") + "/system_stats", timeout=4
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        argv = [str(item) for item in (payload.get("system") or {}).get("argv") or []]
        return [flag for flag in _VramHandoff._COMFY_CPU_OFFLOAD_FLAGS if flag in argv]

    @staticmethod
    def _comfy_free_gb(comfy_url: str) -> float | None:
        """ComfyUI's own view of the VRAM it can allocate, or None.

        For a ComfyUI render this is the authoritative number and the
        driver's is not: after a render ComfyUI keeps a CUDA context and
        allocator pools that nvidia-smi still counts as used, but which
        ComfyUI reuses for the next render without asking the driver for
        anything. Gating that render on driver-free memory demands the same
        capacity twice and refuses a second render that the first one
        proved fits -- observed here as a failure on concept 2 of 6 with
        the reviewer already unloaded and nothing else on the card.
        """
        try:
            with urllib.request.urlopen(
                comfy_url.rstrip("/") + "/system_stats", timeout=4
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        for device in payload.get("devices") or []:
            free = device.get("vram_free")
            if isinstance(free, (int, float)):
                return float(free) / (1024**3)
        return None

    @staticmethod
    def _comfy_reserved_gb(comfy_url: str) -> float:
        """ComfyUI's own --reserve-vram, which is driver headroom it already
        holds back. Studio's safety margin exists for the same purpose, so
        counting both makes the requirement 1.5 GB of headroom on an 8 GB
        card and refuses jobs that fit."""
        try:
            with urllib.request.urlopen(
                comfy_url.rstrip("/") + "/system_stats", timeout=4
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return 0.0
        argv = [str(item) for item in (payload.get("system") or {}).get("argv") or []]
        if "--reserve-vram" in argv:
            index = argv.index("--reserve-vram")
            if index + 1 < len(argv):
                try:
                    return float(argv[index + 1])
                except ValueError:
                    return 0.0
        return 0.0

    @staticmethod
    def _comfy_required(run: StudioRun, workflow: Any) -> float:
        base = _VramHandoff._COMFY_REQUIRED_GB.get(run.concept_backend, 6.0)
        if isinstance(workflow, dict):
            if any(node.get("class_type") == "Hunyuan3Dv2Conditioning" for node in workflow.values()):
                base = _VramHandoff._COMFY_REQUIRED_GB["hunyuan3d"]
            else:
                names = " ".join(
                    str(value)
                    for node in workflow.values()
                    for value in (node.get("inputs") or {}).values()
                    if isinstance(value, str)
                ).lower()
                if "z_image" in names:
                    base = _VramHandoff._COMFY_REQUIRED_GB["z_image_turbo"]
                elif "qwen_image" in names:
                    base = _VramHandoff._COMFY_REQUIRED_GB["qwen_image_2512"]
        return base * _VramHandoff._comfy_mode_fraction(run.comfy_url)

    @classmethod
    def wrap_qwen(cls, inner: Any, run: StudioRun, lease_root: Path) -> Any:
        return cls._Proxy(inner, run, lease_root, "qwen")

    @classmethod
    def wrap_comfy(cls, inner: Any, run: StudioRun, lease_root: Path) -> Any:
        return cls._Proxy(inner, run, lease_root, "comfy")


class StudioCoordinator:
    """Runs one GPU-heavy Studio stage at a time and stops at every human gate."""

    def __init__(
        self,
        store: StudioStore,
        *,
        qwen_factory=None,
        comfy_factory=None,
        worker_executor=None,
        script_runner=None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.store = store
        real_qwen = qwen_factory is None
        real_comfy = comfy_factory is None
        # Whether this coordinator drives real local services or injected
        # fakes. missing_services() needs to know: probing loopback is the
        # right precondition for a real run and meaningless for a test that
        # supplies its own providers.
        self._real_qwen = real_qwen
        self._real_comfy = real_comfy
        self._qwen_factory = qwen_factory or (
            lambda run: StudioQwen(
                base_url=run.localdeploy_url,
                model=run.model,
                spec_strategy=run.spec_strategy,
                timeout_seconds=run.llm_timeout_seconds,
            )
        )
        self._comfy_factory = comfy_factory or (lambda run: StudioComfyClient(run.comfy_url))
        # On a small card the reviewer LLM and the image model cannot both be
        # resident, and D1 alternates between them within a single stage. See
        # _VramHandoff: these wrappers evict the *other* service before each
        # call when run.vram_handoff is on.
        inner_qwen, inner_comfy = self._qwen_factory, self._comfy_factory
        lease_root = self.store.workspace / "scheduler"
        self._qwen_factory = lambda run: (
            _VramHandoff.wrap_qwen(inner_qwen(run), run, lease_root)
            if real_qwen and (run.vram_handoff or run.device_policy != "prefer_gpu")
            else inner_qwen(run)
        )
        self._comfy_factory = lambda run: (
            _VramHandoff.wrap_comfy(inner_comfy(run), run, lease_root)
            if real_comfy and (run.vram_handoff or run.device_policy != "prefer_gpu")
            else inner_comfy(run)
        )
        # D2-D5/D9's typed subprocess worker protocol (blender, trellis2.4b, ...),
        # injectable the same way qwen_factory/comfy_factory are. None (the
        # default) means _execute_worker does what it always has: read
        # config.local.toml and run a real subprocess via WorkerManager. A
        # test supplies a callable(worker_id, request, *, timeout_seconds) ->
        # ExternalWorkerResponse instead, exactly like FakeQwen/FakeComfy.
        self._worker_executor = worker_executor
        # The second, separate subprocess boundary: helper scripts under
        # resources/adapters/ invoked directly (D7's motion chain, D8's surface bake and
        # review, D2's GLB diagnostic) rather than through the typed worker
        # protocol. None means the real subprocess.run. A test supplies a
        # callable(command, *, timeout) -> CompletedProcess-like.
        self._script_runner = script_runner
        # Deterministic provider tests must not depend on, or pay subprocess
        # latency for, the host's GPU tools. Production coordinators use the
        # real service factories and sample at most once per polling interval.
        self._telemetry_enabled = real_qwen or real_comfy
        self._last_gpu_snapshot: dict[str, Any] | None = None
        self._last_gpu_snapshot_at = 0.0
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="text2model-studio")
        self._owns_executor = executor is None
        self._lock = threading.Lock()
        self._jobs: dict[str, Future[None]] = {}
        self._active_comfy: dict[str, ComfyProvider] = {}
        self._stop_requested: set[str] = set()

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, run_id: str) -> bool:
        with self._lock:
            current = self._jobs.get(run_id)
            if current is not None and not current.done():
                return False
            self._stop_requested.discard(run_id)
            self._jobs[run_id] = self._executor.submit(self._drive, run_id)
            return True

    def missing_services(self, settings: dict[str, Any]) -> list[dict[str, str]]:
        """Local services a real run needs that are not answering right now.

        Checked *before* a run is created rather than discovered at D0,
        because a run started without its reviewer cannot reach any gate:
        it fails on its first call and leaves behind a permanently failed
        run that never produced evidence. Returning the remedy alongside
        the name keeps the caller from having to know how each service is
        started.

        Returns an empty list when providers were injected -- a test or an
        embedding caller supplies its own, so loopback readiness says
        nothing about whether that run can proceed.
        """
        wanted: list[tuple[str, str, str, str]] = []
        if self._real_qwen:
            reviewer = str(settings.get("localdeploy_url", "")).rstrip("/")
            if reviewer:
                wanted.append(
                    (
                        "Reviewer model",
                        reviewer,
                        f"{reviewer}/models",
                        "Start Ollama (it serves the reviewer): ollama serve",
                    )
                )
        if self._real_comfy:
            comfy = str(settings.get("comfy_url", "")).rstrip("/")
            if comfy:
                wanted.append(
                    (
                        "ComfyUI",
                        comfy,
                        f"{comfy}/system_stats",
                        "Start ComfyUI: python main.py --listen 127.0.0.1 --port 8188 "
                        "--normalvram --reserve-vram 0.75 --preview-method none",
                    )
                )
        if not wanted:
            return []

        def reachable(url: str) -> bool:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    return response.status == 200
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
            results = list(pool.map(lambda item: reachable(item[2]), wanted))
        return [
            {"name": name, "url": base, "remedy": remedy}
            for (name, base, _probe_url, remedy), ok in zip(wanted, results)
            if not ok
        ]

    def run_to_stop(self, run_id: str) -> None:
        """Advance one run on the calling thread until it reaches its next
        stopping point (a human gate, completion, failure, or block).

        submit() hands the work to this coordinator's background executor
        because its caller -- an HTTP request handler -- must not block. A
        one-shot CLI invocation has no such request thread to free, and
        blocking is exactly what it wants: Ctrl+C should behave like it does
        for any ordinary call, not leave an orphaned background thread that
        keeps the process alive past its own timeout. This runs the same
        _drive() a submitted job runs, just without the executor indirection.
        """
        with self._lock:
            current = self._jobs.get(run_id)
            if current is not None and not current.done():
                raise RuntimeError(f"a Studio job is already running for {run_id}")
            self._stop_requested.discard(run_id)
        self._drive(run_id)

    def submit_manual_qwen_image(
        self,
        run_id: str,
        prompt: str,
        *,
        seed: int | None = None,
    ) -> bool:
        """Render one human-authored D1 candidate through Qwen Image 2512.

        The direct prompt is intentionally an additional *candidate*, not a
        bypass of the Studio contract.  It is preserved as evidence, expanded
        by the LocalDeploy Qwen prompt worker when available, then reviewed by
        the same Qwen gate critic as ordinary candidates.
        """
        prompt = prompt.strip()
        if len(prompt) < 12:
            raise ValueError("the direct Qwen Image prompt must contain at least 12 characters")
        if seed is not None and seed < 0:
            raise ValueError("the optional seed must be zero or a positive integer")
        # Check the gate precondition here, synchronously, so a request that
        # arrives after the gate has already moved on (a stale second browser
        # tab) is refused with an explanation instead of being accepted and
        # then failing inside the worker thread -- where the failure handler
        # would mark the still-waiting D1 gate "failed".
        run = self.store.load(run_id)
        if run.current_stage != "D1" or run.stage("D1").state != "awaiting_review":
            raise ValueError(
                "a direct Qwen Image render is available only while the D1 concept gate awaits review"
            )
        with self._lock:
            current = self._jobs.get(run_id)
            if current is not None and not current.done():
                return False
            self._stop_requested.discard(run_id)
            self._jobs[run_id] = self._executor.submit(
                self._run_manual_qwen_image,
                run_id,
                prompt,
                seed,
            )
            return True

    _MAX_UPLOAD_BYTES = 20 * 1024 * 1024

    def submit_manual_image_upload(self, run_id: str, image_bytes: bytes, filename: str) -> bool:
        """Accept one human-supplied D1 candidate image instead of a generated one.

        Treated as an additional *candidate*, exactly like the direct Qwen
        Image path above: it still has to pass the same deterministic
        chroma/quality gate every generated concept passes, because D2
        depends on that gate's isolated alpha -- not because an upload is
        second-class. An image without a clean, mostly edge-connected green
        background will fail that gate, correctly, the same way a bad
        generated render does, with a specific reason instead of a silent
        bypass; see make_chroma_alpha's own docstring for why this never
        falls back to a learned segmentation model.
        """
        if not image_bytes:
            raise ValueError("no image was uploaded")
        if len(image_bytes) > self._MAX_UPLOAD_BYTES:
            raise ValueError(
                f"uploaded image must be {self._MAX_UPLOAD_BYTES // (1024 * 1024)} MB or smaller"
            )
        try:
            with Image.open(io.BytesIO(image_bytes)) as probe:
                probe.verify()
        except Exception as exc:
            raise ValueError(f"not a readable image file: {type(exc).__name__}: {exc}") from exc
        # Same synchronous precondition check as submit_manual_qwen_image(),
        # for the same reason: refuse a stale request with an explanation
        # instead of accepting it and failing inside the worker thread.
        run = self.store.load(run_id)
        if run.current_stage != "D1" or run.stage("D1").state != "awaiting_review":
            raise ValueError(
                "an uploaded candidate is available only while the D1 concept gate awaits review"
            )
        with self._lock:
            current = self._jobs.get(run_id)
            if current is not None and not current.done():
                return False
            self._stop_requested.discard(run_id)
            self._jobs[run_id] = self._executor.submit(
                self._run_manual_image_upload,
                run_id,
                image_bytes,
                filename,
            )
            return True

    def busy(self, run_id: str) -> bool:
        with self._lock:
            current = self._jobs.get(run_id)
            return current is not None and not current.done()

    def stopping(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._stop_requested

    def _register_comfy(self, run_id: str, comfy: ComfyProvider) -> None:
        with self._lock:
            self._active_comfy[run_id] = comfy

    def _clear_comfy(self, run_id: str) -> None:
        with self._lock:
            self._active_comfy.pop(run_id, None)

    def _was_stopped(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._stop_requested

    def _clear_stop(self, run_id: str) -> None:
        """Retire a stop request once the job it interrupted has ended.

        Without this the flag survives until the *next* submit, so
        stopping() keeps reporting "waiting for the worker to reach a safe
        stop point" at an idle run, and the console contradicts the Resume
        button sitting next to it.
        """
        with self._lock:
            self._stop_requested.discard(run_id)

    def _mark_stopped(self, run_id: str) -> None:
        run = self.store.load(run_id)
        stage = run.stage(run.current_stage)
        stage.state = "blocked"
        stage.error = None
        stage.finished_at = utc_now()
        stage.message = "Stopped by you. Evidence is preserved; use Resume when you are ready to continue."
        run.state = "blocked"
        self.store.event(
            run,
            "human_stopped_job",
            {"stage_id": stage.stage_id, "iteration": stage.iteration},
        )

    def _record_stop_requested(self, run_id: str) -> None:
        run = self.store.load(run_id)
        self.store.event(
            run,
            "human_stop_requested",
            {"stage_id": run.current_stage, "iteration": run.stage(run.current_stage).iteration},
        )

    def stop(self, run_id: str) -> tuple[bool, str]:
        """Interrupt only the currently tracked Studio workflow, never an arbitrary process."""
        with self._lock:
            current = self._jobs.get(run_id)
            if current is None or current.done():
                return False, "No Studio job is running for this asset."
            self._stop_requested.add(run_id)
            queued_cancelled = current.cancel()
            comfy = self._active_comfy.get(run_id)
        self._record_stop_requested(run_id)
        if queued_cancelled:
            # The job never started, so no worker thread will reach the
            # finally that normally retires the flag -- retire it here.
            self._mark_stopped(run_id)
            self._clear_stop(run_id)
            return True, "The queued Studio job was cancelled before it reached ComfyUI."
        if comfy is None:
            return True, "Stop requested. The current Studio step will stop before it launches ComfyUI."
        interrupt = getattr(comfy, "interrupt", None)
        if not callable(interrupt):
            return True, "Stop requested. The active adapter cannot interrupt ComfyUI, so Studio will stop at its next safe boundary."
        try:
            interrupt()
        except Exception as exc:
            return True, f"Stop requested; ComfyUI did not acknowledge the interrupt ({type(exc).__name__}: {exc})."
        return True, "Stop requested; ComfyUI was told to interrupt the active Studio workflow."

    def release_comfy_memory(self, run_id: str) -> tuple[bool, str]:
        """Unload ComfyUI models only when this Studio run is idle."""
        if self.busy(run_id):
            return False, "Stop or wait for the active Studio job before releasing ComfyUI model memory."
        run = self.store.load(run_id)
        comfy = self._comfy_factory(run)
        release = getattr(comfy, "free_memory", None)
        if not callable(release):
            return False, "This ComfyUI adapter does not expose memory release controls."
        try:
            release(unload_models=True, free_memory=True)
        except Exception as exc:
            return False, f"ComfyUI did not release memory: {type(exc).__name__}: {exc}"
        stage = run.stage(run.current_stage)
        stage.message = (
            "ComfyUI models and its execution cache were released. The next render will reload the required models."
        )
        self.store.event(
            run,
            "comfy_memory_released",
            {"stage_id": stage.stage_id, "unload_models": True, "free_memory": True},
        )
        return True, "ComfyUI models and execution cache were released."

    def _drive(self, run_id: str) -> None:
        try:
            while True:
                if self._was_stopped(run_id):
                    self._mark_stopped(run_id)
                    return
                run = self.store.load(run_id)
                stage = self._next_stage(run)
                if stage is None:
                    return
                try:
                    if stage.stage_id == "D0":
                        self._run_d0(run)
                    elif stage.stage_id == "D1":
                        self._run_d1(run)
                    elif stage.stage_id == "D2":
                        self._run_d2(run)
                    elif stage.stage_id == "D3":
                        self._run_d3(run)
                    elif stage.stage_id == "D4":
                        self._run_d4(run)
                    elif stage.stage_id == "D5":
                        self._run_d5(run)
                    elif stage.stage_id == "D6":
                        self._run_d6(run)
                    elif stage.stage_id == "D7":
                        self._run_d7(run)
                    elif stage.stage_id == "D8":
                        self._run_d8(run)
                    elif stage.stage_id == "D9":
                        self._run_d9(run)
                    elif stage.stage_id == "D10":
                        self._run_d10(run)
                    else:
                        stage.state = "blocked"
                        stage.message = (
                            "The Studio control contract is ready, but this production adapter has not yet been "
                            "connected to the generic character/equipment pipeline."
                        )
                        run.state = "blocked"
                        run.current_stage = stage.stage_id
                        self.store.event(
                            run,
                            "stage_blocked",
                            {"stage_id": stage.stage_id, "reason": stage.message},
                        )
                        return
                except Exception as exc:
                    if self._schedule_automatic_retry(run_id, exc):
                        continue
                    raise
                if self._was_stopped(run_id):
                    self._mark_stopped(run_id)
                    return
                if self.store.load(run_id).state == "blocked":
                    return
        except Exception as exc:
            if self._was_stopped(run_id):
                self._mark_stopped(run_id)
                return
            run = self.store.load(run_id)
            stage = run.stage(run.current_stage)
            stage.state = "failed"
            stage.error = f"{type(exc).__name__}: {exc}"
            stage.message = "Stage failed. Fix the dependency or use Resume to retry without losing history."
            stage.finished_at = utc_now()
            run.state = "failed"
            self.store.event(
                run,
                "stage_failed",
                {"stage_id": stage.stage_id, "error": stage.error},
            )
        finally:
            self._clear_comfy(run_id)
            self._clear_stop(run_id)

    def _schedule_automatic_retry(self, run_id: str, error: Exception) -> bool:
        """Retry only transient failures or OOMs with a changed memory plan."""

        run = self.store.load(run_id)
        stage = run.stage(run.current_stage)
        if stage.retry_attempt >= run.automatic_retry_limit:
            return False
        detail = f"{type(error).__name__}: {error}"
        lowered = detail.lower()
        out_of_memory = any(
            token in lowered
            for token in (
                "out of memory",
                "cuda oom",
                "cudnn_status_alloc_failed",
                "memory admission denied",
                "gpu admission denied",
            )
        )
        transient = isinstance(error, StudioComfyError) and any(
            token in lowered for token in ("timed out", "urlerror", "connection", "temporarily", "reset")
        )
        if not out_of_memory and not transient:
            return False
        overrides: dict[str, Any] = {}
        if out_of_memory:
            retry_number = stage.retry_attempt + 1
            if stage.stage_id == "D1":
                scale = 0.8**retry_number
                overrides = {
                    "concept_width": max(512, int(run.concept_width * scale) // 64 * 64),
                    "concept_height": max(640, int(run.concept_height * scale) // 64 * 64),
                    "concept_candidates": max(2, run.concept_candidates - retry_number),
                }
            elif stage.stage_id == "D2":
                current = self._stage_settings(run, "D2")
                overrides = {
                    "octree_resolution": max(
                        128,
                        int(current.get("octree_resolution", 256)) // (2**retry_number),
                    ),
                    "steps": max(12, int(current.get("steps", 20)) - 4 * retry_number),
                }
            elif stage.stage_id in {"D3", "D4", "D9"}:
                current = self._stage_settings(run, stage.stage_id)
                overrides = {
                    "render_size": max(
                        128,
                        int(current.get("render_size", 512)) // (2**retry_number),
                    )
                }
            else:
                return False
        stage.retry_attempt += 1
        stage.retry_reason = detail
        stage.pending_overrides = overrides
        stage.state = "failed"
        stage.error = detail
        stage.message = (
            f"Automatic retry {stage.retry_attempt} of {run.automatic_retry_limit} scheduled"
            + (" with a smaller memory plan." if overrides else " after a transient service failure.")
        )
        run.state = "running"
        self.store.event(
            run,
            "automatic_retry_scheduled",
            {
                "stage_id": stage.stage_id,
                "attempt": stage.retry_attempt,
                "reason": detail,
                "changed_parameters": overrides,
            },
        )
        return True

    @staticmethod
    def _next_stage(run: StudioRun):
        for stage in run.stages:
            if stage.state == "skipped":
                continue
            if stage.state in {"pending", "queued", "rejected", "failed", "blocked"}:
                preceding = run.stages[: run.stages.index(stage)]
                if any(item.state not in {"approved", "skipped"} for item in preceding):
                    return None
                return stage
            if stage.state == "awaiting_review":
                return None
        run.state = "completed"
        run.current_stage = "D10"
        return None

    def _begin(self, run: StudioRun, stage_id: str, message: str) -> dict[str, Any]:
        """Start one attempt of a stage and return the human-supplied parameter
        overrides that apply to *this* attempt.

        A retry/edit decision parks its overrides on the stage; they are
        consumed here (exactly once, then cleared) so a later automatic
        iteration does not silently keep re-applying a one-off correction.
        """
        stage = run.stage(stage_id)
        retrying_pre_output_failure = stage.state == "failed" and not any(
            item.metrics.get("iteration") == stage.iteration for item in stage.evidence
        )
        stage.state = "running"
        stage.progress = 0.01
        stage.progress_phase = "starting"
        stage.progress_current = 0
        stage.progress_total = 0
        stage.progress_unit = ""
        if not retrying_pre_output_failure:
            stage.iteration += 1
        stage.started_at = utc_now()
        stage.finished_at = None
        stage.error = None
        stage.message = message
        run.state = "running"
        run.current_stage = stage_id
        overrides = dict(stage.pending_overrides)
        stage.pending_overrides = {}
        self.store.event(
            run,
            "stage_started",
            {"stage_id": stage_id, "iteration": stage.iteration, "message": message},
        )
        if overrides:
            self.store.event(
                run,
                "stage_overrides_applied",
                {"stage_id": stage_id, "iteration": stage.iteration, "overrides": overrides},
            )
        return overrides

    def _progress(
        self,
        run: StudioRun,
        stage_id: str,
        value: float,
        message: str,
        *,
        phase: str | None = None,
        current: int | None = None,
        total: int | None = None,
        unit: str | None = None,
    ) -> None:
        stage = run.stage(stage_id)
        stage.progress = value
        stage.message = message
        if phase is not None:
            stage.progress_phase = phase
        if current is not None:
            stage.progress_current = current
        if total is not None:
            stage.progress_total = total
        if unit is not None:
            stage.progress_unit = unit
        snapshot = self._last_gpu_snapshot
        if self._telemetry_enabled and time.monotonic() - self._last_gpu_snapshot_at >= 0.75:
            snapshot = gpu_memory_snapshot()
            self._last_gpu_snapshot = snapshot
            self._last_gpu_snapshot_at = time.monotonic()
        device = ((snapshot or {}).get("device") or {})
        if isinstance(device.get("used_gb"), (int, float)):
            stage.gpu_used_gb = float(device["used_gb"])
        if isinstance(device.get("free_gb"), (int, float)):
            stage.gpu_free_gb = float(device["free_gb"])
        if isinstance(device.get("total_gb"), (int, float)):
            stage.gpu_total_gb = float(device["total_gb"])
        self.store.save(run)

    def _run_manual_qwen_image(self, run_id: str, human_prompt: str, seed: int | None) -> None:
        """Run the direct-prompt Qwen Image 2512 path from the D1 human gate."""
        try:
            run = self.store.load(run_id)
            stage = run.stage("D1")
            if run.current_stage != "D1" or stage.state != "awaiting_review":
                # submit_manual_qwen_image() already checked this; losing the
                # race with a decision recorded in between is not a stage
                # failure, so record it and leave the gate exactly as it is.
                self.store.event(
                    run,
                    "manual_qwen_image_refused",
                    {"stage_id": run.current_stage, "reason": "the D1 concept gate is no longer awaiting review"},
                )
                return
            if run.spec is None:
                raise RuntimeError("D1 requires the compiled D0 specification")

            self._begin(run, "D1", "Preparing your direct prompt for Qwen Image 2512.")
            stage = run.stage("D1")
            iteration = stage.iteration
            attempt_root = (
                self.store.run_root(run.run_id)
                / "D1_concept"
                / f"iteration-{iteration:02d}"
                / "manual_qwen_image"
            )
            actual_seed = seed if seed is not None else secrets.randbelow(2_147_483_647)
            direct_positive, direct_negative = _whole_asset_prompt_guard(
                run.spec,
                human_prompt,
                (
                    "pixel art, retro sprites, 8-bit graphics, voxel blocks, hard black outlines, chibi, "
                    "duplicate people, extra limbs, floating equipment, text, logos, watermark"
                ),
                defer_shield=False,
            )
            direct_positive, direct_negative = _geometry_ready_backdrop_guard(
                direct_positive,
                direct_negative,
            )
            # ConceptPlan gives the existing Qwen prompt rewriter the same
            # typed contract as a pipeline-authored candidate.  Pad terse but
            # valid human prompts with neutral rendering requirements rather
            # than silently rejecting a useful instruction.
            if len(direct_positive) < 80:
                direct_positive += " Render one complete original asset in a neutral studio presentation."
            plan = ConceptPlan(
                positive_prompt=direct_positive,
                negative_prompt=direct_negative,
                seeds=[actual_seed, actual_seed + 1],
                rationale="A human-authored direct Qwen Image candidate at the D1 review gate.",
            )
            qwen = self._qwen_factory(run)
            instruction = None
            instruction_error = ""
            effective_prompt = plan.positive_prompt
            effective_negative = plan.negative_prompt
            rewriter = getattr(qwen, "image_edit_instruction", None)
            if callable(rewriter):
                try:
                    instruction = rewriter(
                        run.spec,
                        stage,
                        plan,
                        source_handling="new_text_to_image",
                    )
                    effective_prompt = instruction.prompt
                    effective_negative = instruction.negative_prompt
                    effective_prompt, effective_negative = _geometry_ready_backdrop_guard(
                        effective_prompt,
                        effective_negative,
                    )
                except Exception as exc:
                    instruction_error = f"{type(exc).__name__}: {exc}"

            request_path = attempt_root / "manual_qwen_image_request.json"
            _write_json(
                request_path,
                {
                    "schema_version": 1,
                    "human_prompt": human_prompt,
                    "seed": actual_seed,
                    "backend": "qwen_image_2512",
                    "source_handling": "new_text_to_image",
                    "qwen_instruction": instruction.model_dump(mode="json") if instruction else None,
                    "qwen_instruction_error": instruction_error or None,
                },
            )
            self.store.evidence(
                run,
                "D1",
                request_path,
                evidence_id=f"d1-i{iteration:02d}-manual-qwen-image-request",
                label=f"Your direct Qwen Image prompt, iteration {iteration}",
                media_type="application/json",
                metrics={
                    "iteration": iteration,
                    "selectable": False,
                    "role": "human_direct_prompt",
                    "workflow_strategy": "qwen_image_2512_manual_prompt_v1",
                    "seed": actual_seed,
                },
            )
            self._progress(run, "D1", 0.16, "Checking that the native Qwen Image 2512 model is available.")
            comfy = self._comfy_factory(run)
            self._register_comfy(run_id, comfy)
            models_provider = getattr(comfy, "models", None)
            installed_diffusion_models = models_provider("diffusion_models") if callable(models_provider) else []
            installed_text_encoders = models_provider("text_encoders") if callable(models_provider) else []
            installed_vaes = models_provider("vae") if callable(models_provider) else []
            required_models = (
                ("qwen_image_2512_fp8_e4m3fn.safetensors", installed_diffusion_models),
                ("qwen_2.5_vl_7b_fp8_scaled.safetensors", installed_text_encoders),
                ("qwen_image_vae.safetensors", installed_vaes),
            )
            if not all(required in installed for required, installed in required_models):
                raise RuntimeError(
                    "Direct Qwen Image requires the Qwen Image 2512 model, text encoder, and VAE in ComfyUI. "
                    "Run resources/adapters/install_qwen_image_edit_models.py --profile image-2512."
                )
            profile_path = attempt_root / "workflow_profile.json"
            _write_json(
                profile_path,
                {
                    "schema_version": 1,
                    "strategy": "qwen_image_2512_manual_prompt_v1",
                    "backend": "qwen_image_2512",
                    "seed": actual_seed,
                    "human_prompt_preserved": True,
                    "qwen_prompt_rewritten": instruction is not None,
                    "qwen_prompt_rewrite_error": instruction_error or None,
                    "effective_positive_prompt": effective_prompt,
                    "effective_negative_prompt": effective_negative,
                },
            )
            self.store.evidence(
                run,
                "D1",
                profile_path,
                evidence_id=f"d1-i{iteration:02d}-manual-qwen-image-profile",
                label=f"Direct Qwen Image workflow profile, iteration {iteration}",
                media_type="application/json",
                metrics={
                    "iteration": iteration,
                    "selectable": False,
                    "role": "workflow_profile",
                    "workflow_strategy": "qwen_image_2512_manual_prompt_v1",
                },
            )
            self._progress(run, "D1", 0.28, f"Qwen Image 2512 is rendering your direct candidate (seed {actual_seed}).")
            outputs = comfy.generate(
                workflow=qwen_image_2512_workflow(
                    prompt=effective_prompt,
                    negative_prompt=effective_negative,
                    seed=actual_seed,
                    prefix=f"Text2ModelStudio/{run.run_id}/manual/i{iteration:02d}",
                    width=run.concept_width,
                    height=run.concept_height,
                    tiled_vae=run.vae_tiling,
                ),
                destination=attempt_root / "candidate-1",
                timeout_seconds=1200,
            )
            if self._was_stopped(run_id):
                self._mark_stopped(run_id)
                return
            if not outputs:
                raise RuntimeError("Qwen Image 2512 returned no output image")
            image = outputs[0]
            evidence_id = f"d1-i{iteration:02d}-candidate-qwen-image-2512-manual-01"
            alpha_preview = attempt_root / "candidate-1" / "geometry_ready_rgba.png"
            alpha_error = ""
            try:
                alpha_metrics = make_chroma_alpha(image, alpha_preview)
            except Exception as exc:
                alpha_metrics = {"meaningful_alpha": False}
                alpha_error = f"{type(exc).__name__}: {exc}"
                alpha_preview = None
            quality = assess_concept_image(
                image,
                alpha_preview,
                minimum_score=run.concept_min_quality_score,
            )
            equipment = (
                check_equipment_conformance(qwen, image, run.spec)
                if callable(getattr(qwen, "visual_presence", None))
                else EquipmentConformanceReport(conforms=True)
            )
            if alpha_preview is not None:
                self.store.evidence(
                    run,
                    "D1",
                    alpha_preview,
                    evidence_id=f"{evidence_id}-geometry-ready-alpha",
                    label="Deterministic geometry-ready alpha for your candidate",
                    media_type="image/png",
                    metrics={
                        **alpha_metrics,
                        "iteration": iteration,
                        "selectable": False,
                        "role": "geometry_ready_alpha",
                        "source_evidence_id": evidence_id,
                    },
                )
            metrics = _image_metrics(image)
            metrics.update(
                {
                    "seed": actual_seed,
                    "iteration": iteration,
                    "selectable": equipment.conforms and quality.hard_requirements_satisfied,
                    "operation_id": "human_direct_qwen_image_prompt",
                    "workflow_strategy": "qwen_image_2512_manual_prompt_v1",
                    "concept_backend": "qwen_image_2512",
                    "human_prompt": True,
                    "qwen_prompt_rewritten": instruction is not None,
                    "equipment_conformance_ok": equipment.conforms,
                    "equipment_conformance_violations": "; ".join(equipment.violations),
                    **quality.metrics,
                    "quality_reasons": "; ".join(quality.reasons),
                    "alpha_error": alpha_error,
                }
            )
            self.store.evidence(
                run,
                "D1",
                image,
                evidence_id=evidence_id,
                label=f"Your direct Qwen Image candidate, iteration {iteration}",
                media_type="image/png",
                metrics=metrics,
            )
            candidates = [(evidence_id, image, metrics)]
            comparison_board = _concept_comparison_board(
                self.store,
                run,
                stage,
                candidates,
                attempt_root / "previous_vs_direct_candidate.png",
            )
            self.store.evidence(
                run,
                "D1",
                comparison_board,
                evidence_id=f"d1-i{iteration:02d}-manual-qwen-image-comparison-board",
                label=f"Direct Qwen Image comparison, iteration {iteration}",
                media_type="image/png",
                metrics={"iteration": iteration, "selectable": False},
            )
            self._progress(run, "D1", 0.82, "Qwen critic is reviewing your direct candidate against the typed contract.")
            try:
                review = qwen.review_concepts(
                    run.spec,
                    stage,
                    candidates,
                    comparison_board=comparison_board,
                )
            except Exception as exc:
                review = StudioQwenReview(
                    review_id=f"d1.manual-qwen.unavailable-{iteration:02d}",
                    stage_id="D1",
                    iteration=iteration,
                    summary="The direct Qwen Image candidate rendered, but the Qwen critic did not complete.",
                    issues=[f"Qwen critic error: {type(exc).__name__}: {exc}"],
                    candidate_ranking=[evidence_id],
                    recommended_evidence_id=None,
                    recommended_changes=["Use your review comment to direct the next candidate."],
                    confidence=0.0,
                    hard_requirements_satisfied=False,
                    request_human_review=True,
                )
            stage.qwen_reviews.append(review)
            review_path = attempt_root / "qwen_review.json"
            _write_json(review_path, review.model_dump(mode="json"))
            self.store.evidence(
                run,
                "D1",
                review_path,
                evidence_id=f"d1-i{iteration:02d}-manual-qwen-image-review",
                label=f"Qwen review of your direct candidate, iteration {iteration}",
                media_type="application/json",
                metrics={"iteration": iteration, "selectable": False, "confidence": review.confidence},
            )
            stage.metrics = {
                "iteration": iteration,
                "candidate_count": 1,
                "qwen_confidence": review.confidence,
                "recommended_evidence_id": review.recommended_evidence_id or "",
                "hard_requirements_satisfied": review.hard_requirements_satisfied,
                "workflow_strategy": "qwen_image_2512_manual_prompt_v1",
                "human_prompt": True,
            }
            stage.progress = 1
            stage.finished_at = utc_now()
            stage.state = "awaiting_review"
            stage.message = "Your direct Qwen Image candidate and Qwen review are ready for approval or rejection."
            run.state = "awaiting_review"
            self.store.event(
                run,
                "manual_qwen_image_ready",
                {
                    "stage_id": "D1",
                    "iteration": iteration,
                    "evidence_id": evidence_id,
                    "recommended_evidence_id": review.recommended_evidence_id,
                },
            )
        except Exception as exc:
            if self._was_stopped(run_id):
                self._mark_stopped(run_id)
                return
            run = self.store.load(run_id)
            stage = run.stage(run.current_stage)
            stage.state = "failed"
            stage.error = f"{type(exc).__name__}: {exc}"
            stage.message = "Direct Qwen Image render failed. Fix the dependency or use Resume without losing history."
            stage.finished_at = utc_now()
            run.state = "failed"
            self.store.event(
                run,
                "manual_qwen_image_failed",
                {"stage_id": stage.stage_id, "error": stage.error},
            )
        finally:
            self._clear_comfy(run_id)
            self._clear_stop(run_id)

    def _run_manual_image_upload(self, run_id: str, image_bytes: bytes, filename: str) -> None:
        """Run the human-uploaded D1 candidate path from the D1 human gate.

        Mirrors _run_manual_qwen_image above with no ComfyUI render: the
        uploaded bytes stand in for what a backend would have generated, and
        everything downstream of that -- chroma alpha, deterministic
        quality, equipment conformance, evidence, comparison board, Qwen
        review -- is the same code every other D1 candidate runs through.
        """
        try:
            run = self.store.load(run_id)
            stage = run.stage("D1")
            if run.current_stage != "D1" or stage.state != "awaiting_review":
                self.store.event(
                    run,
                    "manual_image_upload_refused",
                    {"stage_id": run.current_stage, "reason": "the D1 concept gate is no longer awaiting review"},
                )
                return
            if run.spec is None:
                raise RuntimeError("D1 requires the compiled D0 specification")

            self._begin(run, "D1", "Preparing your uploaded candidate.")
            stage = run.stage("D1")
            iteration = stage.iteration
            attempt_root = (
                self.store.run_root(run.run_id)
                / "D1_concept"
                / f"iteration-{iteration:02d}"
                / "manual_upload"
            )
            image = attempt_root / "candidate-1" / "upload.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(io.BytesIO(image_bytes)) as uploaded:
                    uploaded.convert("RGB").save(image, format="PNG")
            except Exception as exc:
                raise ValueError(f"could not decode the uploaded image: {type(exc).__name__}: {exc}") from exc

            self._progress(run, "D1", 0.35, "Isolating the subject from its background.")
            evidence_id = f"d1-i{iteration:02d}-candidate-upload-01"
            alpha_preview = attempt_root / "candidate-1" / "geometry_ready_rgba.png"
            alpha_error = ""
            try:
                alpha_metrics = make_chroma_alpha(image, alpha_preview)
            except Exception as exc:
                alpha_metrics = {"meaningful_alpha": False}
                alpha_error = f"{type(exc).__name__}: {exc}"
                alpha_preview = None
            quality = assess_concept_image(
                image,
                alpha_preview,
                minimum_score=run.concept_min_quality_score,
            )
            qwen = self._qwen_factory(run)
            equipment = (
                check_equipment_conformance(qwen, image, run.spec)
                if callable(getattr(qwen, "visual_presence", None))
                else EquipmentConformanceReport(conforms=True)
            )
            if alpha_preview is not None:
                self.store.evidence(
                    run,
                    "D1",
                    alpha_preview,
                    evidence_id=f"{evidence_id}-geometry-ready-alpha",
                    label="Deterministic geometry-ready alpha for your uploaded candidate",
                    media_type="image/png",
                    metrics={
                        **alpha_metrics,
                        "iteration": iteration,
                        "selectable": False,
                        "role": "geometry_ready_alpha",
                        "source_evidence_id": evidence_id,
                    },
                )
            metrics = _image_metrics(image)
            metrics.update(
                {
                    "iteration": iteration,
                    "selectable": equipment.conforms and quality.hard_requirements_satisfied,
                    "operation_id": "human_uploaded_image",
                    "workflow_strategy": "human_upload_v1",
                    "concept_backend": "human_upload",
                    "human_uploaded": True,
                    "original_filename": filename[:200],
                    "equipment_conformance_ok": equipment.conforms,
                    "equipment_conformance_violations": "; ".join(equipment.violations),
                    **quality.metrics,
                    "quality_reasons": "; ".join(quality.reasons),
                    "alpha_error": alpha_error,
                }
            )
            self.store.evidence(
                run,
                "D1",
                image,
                evidence_id=evidence_id,
                label=f"Your uploaded candidate, iteration {iteration}",
                media_type="image/png",
                metrics=metrics,
            )
            candidates = [(evidence_id, image, metrics)]
            comparison_board = _concept_comparison_board(
                self.store,
                run,
                stage,
                candidates,
                attempt_root / "previous_vs_uploaded_candidate.png",
            )
            self.store.evidence(
                run,
                "D1",
                comparison_board,
                evidence_id=f"d1-i{iteration:02d}-manual-upload-comparison-board",
                label=f"Uploaded candidate comparison, iteration {iteration}",
                media_type="image/png",
                metrics={"iteration": iteration, "selectable": False},
            )
            self._progress(run, "D1", 0.82, "Qwen critic is reviewing your uploaded candidate against the typed contract.")
            try:
                review = qwen.review_concepts(
                    run.spec,
                    stage,
                    candidates,
                    comparison_board=comparison_board,
                )
            except Exception as exc:
                review = StudioQwenReview(
                    review_id=f"d1.manual-upload.unavailable-{iteration:02d}",
                    stage_id="D1",
                    iteration=iteration,
                    summary="Your uploaded candidate was processed, but the Qwen critic did not complete.",
                    issues=[f"Qwen critic error: {type(exc).__name__}: {exc}"],
                    candidate_ranking=[evidence_id],
                    recommended_evidence_id=None,
                    recommended_changes=["Use your review comment to direct the next candidate."],
                    confidence=0.0,
                    hard_requirements_satisfied=False,
                    request_human_review=True,
                )
            stage.qwen_reviews.append(review)
            review_path = attempt_root / "qwen_review.json"
            _write_json(review_path, review.model_dump(mode="json"))
            self.store.evidence(
                run,
                "D1",
                review_path,
                evidence_id=f"d1-i{iteration:02d}-manual-upload-review",
                label=f"Qwen review of your uploaded candidate, iteration {iteration}",
                media_type="application/json",
                metrics={"iteration": iteration, "selectable": False, "confidence": review.confidence},
            )
            stage.metrics = {
                "iteration": iteration,
                "candidate_count": 1,
                "qwen_confidence": review.confidence,
                "recommended_evidence_id": review.recommended_evidence_id or "",
                "hard_requirements_satisfied": review.hard_requirements_satisfied,
                "workflow_strategy": "human_upload_v1",
                "human_uploaded": True,
            }
            stage.progress = 1
            stage.finished_at = utc_now()
            stage.state = "awaiting_review"
            stage.message = "Your uploaded candidate and Qwen review are ready for approval or rejection."
            run.state = "awaiting_review"
            self.store.event(
                run,
                "manual_image_upload_ready",
                {
                    "stage_id": "D1",
                    "iteration": iteration,
                    "evidence_id": evidence_id,
                    "recommended_evidence_id": review.recommended_evidence_id,
                },
            )
        except Exception as exc:
            if self._was_stopped(run_id):
                self._mark_stopped(run_id)
                return
            run = self.store.load(run_id)
            stage = run.stage(run.current_stage)
            stage.state = "failed"
            stage.error = f"{type(exc).__name__}: {exc}"
            stage.message = "Your uploaded candidate could not be processed. Fix the issue and use Resume without losing history."
            stage.finished_at = utc_now()
            run.state = "failed"
            self.store.event(
                run,
                "manual_image_upload_failed",
                {"stage_id": stage.stage_id, "error": stage.error},
            )
        finally:
            self._clear_stop(run_id)

    def _run_d0(self, run: StudioRun) -> None:
        self._begin(run, "D0", "Qwen is compiling the description into a production contract.")
        qwen = self._qwen_factory(run)
        spec = qwen.compile_spec(run.compilation_brief())
        run.spec = spec
        run.title = spec.title
        root = self.store.run_root(run.run_id)
        spec_path = root / "D0_brief" / "asset_spec.json"
        _write_json(spec_path, spec.model_dump(mode="json"))
        self.store.evidence(
            run,
            "D0",
            spec_path,
            evidence_id="d0-asset-spec",
            label="Qwen-compiled asset specification",
            media_type="application/json",
            metrics={
                "equipment_count": len(spec.equipment),
                "animation_count": len(spec.animations),
                "locked_feature_count": len(spec.locked_features),
            },
        )
        stage = run.stage("D0")
        # Recompute applicability from scratch. D0 can run a second time
        # after a rollback to it, against a freshly compiled -- and possibly
        # differently shaped -- spec. StudioStore._invalidate_from
        # deliberately leaves a contract-excluded stage skipped, so unless
        # the previous contract's exclusions are lifted here, a chair that
        # is recompiled as an animated creature could never regain its
        # skeleton, rig, skinning, and motion stages.
        for item in run.stages:
            if item.stage_id == "D0" or item.applicable:
                continue
            item.applicable = True
            item.state = "pending"
            item.progress = 0
            item.error = None
            item.message = "Reopened while the asset contract is recompiled."
        skip_reasons: dict[str, str] = {}
        if spec.asset_kind == "material":
            skip_reasons.update(
                {stage_id: "Material assets do not require geometry or articulation." for stage_id in ("D2", "D3", "D4", "D5", "D6", "D7")}
            )
        elif spec.behavior == "static":
            skip_reasons.update(
                {stage_id: "Static assets do not require skeleton, rig, deformation, or motion." for stage_id in ("D4", "D5", "D6", "D7")}
            )
        elif spec.behavior == "rigid_articulated":
            skip_reasons["D6"] = "Rigid articulated assets do not require deforming skin weights."
            if not spec.animations:
                skip_reasons["D7"] = "No rigid motion clips were requested."
        elif spec.behavior == "simulated":
            skip_reasons.update(
                {stage_id: "Simulated assets use a simulation contract rather than a skeletal rig." for stage_id in ("D4", "D5", "D6")}
            )
        for skipped_id, reason in skip_reasons.items():
            skipped = run.stage(skipped_id)
            skipped.applicable = False
            skipped.state = "skipped"
            skipped.progress = 1
            skipped.message = reason
        stage.metrics = {
            "asset_kind": spec.asset_kind,
            "behavior": spec.behavior,
            "equipment_count": len(spec.equipment),
            "animation_count": len(spec.animations),
            "explicit_right_weapon": any(
                item.category == "weapon" and item.side == "right" for item in spec.equipment
            ),
            "explicit_left_shield": any(
                item.category == "shield" and item.side == "left" for item in spec.equipment
            ),
        }
        stage.progress = 1
        stage.state = "approved"
        stage.finished_at = utc_now()
        stage.message = "Description compiled into a typed asset contract and deterministic constraints passed."
        self.store.event(run, "automatic_gate_passed", {"stage_id": "D0", "metrics": stage.metrics})

    def _run_d1(self, run: StudioRun) -> None:
        stage = run.stage("D1")
        structural_iterations = {
            int(item.metrics.get("iteration", 0))
            for item in stage.evidence
            if str(item.metrics.get("workflow_strategy", "")).endswith(("_v2", "_v3"))
        }
        qwen_image_iterations = {
            int(item.metrics.get("iteration", 0))
            for item in stage.evidence
            if str(item.metrics.get("workflow_strategy", "")).startswith("qwen_image_")
        }
        qwen_retry_selected = run.concept_backend.startswith("qwen_image_") or (
            run.concept_backend == "auto" and bool(qwen_image_iterations)
        )
        retry_budget_exhausted = (
            len(qwen_image_iterations) >= 3 if qwen_retry_selected else len(structural_iterations) >= 2
        )
        if stage.iteration >= 6 and stage.state == "rejected" and retry_budget_exhausted:
            stage.state = "blocked"
            stage.message = (
                "The bounded concept-backend retry budget was rejected after the original prompt-only attempts. "
                "The evidence is preserved for a human strategy decision instead of spending more GPU time."
            )
            run.state = "blocked"
            self.store.event(
                run,
                "iteration_budget_exhausted",
                {
                    "stage_id": "D1",
                    "workflow_strategy": "qwen_image_2512" if qwen_retry_selected else "pose_guided_lora_v3",
                },
            )
            return
        stage_overrides = self._begin(
            run, "D1", "Qwen is planning the next two concept candidates from the full history."
        )
        if run.spec is None:
            raise RuntimeError("D1 requires the compiled D0 specification")
        qwen = self._qwen_factory(run)
        correction: ConceptCorrectionPlan | None = None
        is_rejected_retry = awaiting_correction(stage)
        if is_rejected_retry:
            all_selectable = [
                item
                for item in stage.evidence
                if item.media_type.startswith("image/")
                and item.metrics.get("selectable") is not False
                and "candidate" in item.evidence_id
            ]
            latest_evidence_iteration = max(
                (int(item.metrics.get("iteration", 0)) for item in all_selectable),
                default=0,
            )
            visible_candidates = [
                item
                for item in all_selectable
                if int(item.metrics.get("iteration", 0)) == latest_evidence_iteration
            ]
            latest_rejection = latest_correction(stage)
            visible_ids = {item.evidence_id for item in visible_candidates}
            if latest_rejection and latest_rejection.selected_evidence_id:
                visible_ids.add(latest_rejection.selected_evidence_id)
            candidate_ids = sorted(visible_ids)
            comparison_item = next(
                (
                    item
                    for item in reversed(stage.evidence)
                    if item.evidence_id.endswith("comparison-board")
                ),
                None,
            )
            comparison_path = (
                self.store.artifact_path(run.run_id, comparison_item.relative_path)
                if comparison_item is not None
                else None
            )
            if comparison_path is None and visible_candidates:
                comparison_path = _concept_comparison_board(
                    self.store,
                    run,
                    stage,
                    [
                        (
                            item.evidence_id,
                            self.store.artifact_path(run.run_id, item.relative_path),
                            dict(item.metrics),
                        )
                        for item in visible_candidates
                    ],
                    self.store.run_root(run.run_id)
                    / "D1_concept"
                    / f"correction_history_before_i{stage.iteration:02d}.png",
                )
            correction = qwen.concept_correction_plan(
                run.spec,
                stage,
                candidate_ids,
                comparison_board=comparison_path,
            )
            plan: ConceptPlan | ConceptCorrectionPlan = correction
        else:
            plan = qwen.concept_plan(run.spec, stage)
        # A human "retry"/"edit" may pin the first seed for this one attempt, so
        # a candidate they liked can be re-rolled deterministically instead of
        # accepting whatever Qwen proposes next. The second seed stays Qwen's so
        # the attempt still offers a genuine alternative to compare against.
        pinned_seed = stage_overrides.get("seed")
        if pinned_seed is not None:
            # Shape already enforced at decision time by
            # studio_models.validate_stage_overrides, so the human got an
            # immediate error rather than an asynchronously failed stage.
            validate_stage_overrides(stage_overrides)
            other = next((item for item in plan.seeds if item != pinned_seed), pinned_seed + 1)
            plan.seeds = [pinned_seed, other]
        candidate_budget = int(stage_overrides.get("concept_candidates", run.concept_candidates))
        concept_width = int(stage_overrides.get("concept_width", run.concept_width))
        concept_height = int(stage_overrides.get("concept_height", run.concept_height))
        candidate_seeds = list(plan.seeds)
        while len(candidate_seeds) < candidate_budget:
            # Golden-ratio Weyl sequence: deterministic, non-repeating within
            # this bounded list, and far enough apart to avoid adjacent-seed
            # correlation without asking the small planner for a larger JSON.
            candidate = (candidate_seeds[-1] + 0x9E3779B97F4A7C15) % (2**63 - 1)
            if candidate not in candidate_seeds:
                candidate_seeds.append(candidate)
        attempt_root = self.store.run_root(run.run_id) / "D1_concept" / f"iteration-{stage.iteration:02d}"
        _write_json(
            attempt_root / "qwen_plan.json",
            {
                "plan": plan.model_dump(mode="json"),
                "mode": "targeted_correction" if correction else "new_generation",
                "history_iteration": stage.iteration,
            },
        )
        self._progress(run, "D1", 0.12, "Qwen plan is ready; selecting the installed ComfyUI concept backend.")
        comfy = self._comfy_factory(run)
        self._register_comfy(run.run_id, comfy)
        controlnet_provider = getattr(comfy, "controlnets", None)
        installed_controlnets = controlnet_provider() if callable(controlnet_provider) else []
        models_provider = getattr(comfy, "models", None)
        installed_loras = models_provider("loras") if callable(models_provider) else []
        installed_diffusion_models = models_provider("diffusion_models") if callable(models_provider) else []
        installed_text_encoders = models_provider("text_encoders") if callable(models_provider) else []
        installed_vaes = models_provider("vae") if callable(models_provider) else []
        qwen_image_2512_models_ready = all(
            required in models
            for required, models in (
                ("qwen_image_2512_fp8_e4m3fn.safetensors", installed_diffusion_models),
                ("qwen_2.5_vl_7b_fp8_scaled.safetensors", installed_text_encoders),
                ("qwen_image_vae.safetensors", installed_vaes),
            )
        )
        if run.concept_backend in {"qwen_image_2512", "qwen_image_edit_2511"} and not qwen_image_2512_models_ready:
            raise RuntimeError(
                "Qwen Image 2512 was selected for concept generation but its required ComfyUI model files are missing. "
                "Run resources/adapters/install_qwen_image_edit_models.py --profile image-2512 or select the SDXL backend."
            )
        z_image_turbo_models_ready = all(
            required in models
            for required, models in (
                ("z_image_turbo_int8_convrot.safetensors", installed_diffusion_models),
                ("qwen_3_4b_fp8_mixed.safetensors", installed_text_encoders),
                ("ae.safetensors", installed_vaes),
            )
        )
        if run.concept_backend == "z_image_turbo" and not z_image_turbo_models_ready:
            raise RuntimeError(
                "Z-Image Turbo was selected for concept generation but its required ComfyUI model files are missing. "
                "Run resources/adapters/install_qwen_image_edit_models.py --profile z-image-turbo or select another backend."
            )
        # "auto" prefers Z-Image Turbo over Qwen-Image-2512 when both are
        # installed. Measured on an 8 GB card, one 768x1024 candidate costs
        # ~48 s on Z-Image against ~605 s on Qwen, for output in the same
        # stylistic family with the same equipment fidelity. Qwen is the
        # better single image and stays selectable for that -- but D1 renders
        # `concept_candidates` per iteration and re-renders the whole set on
        # every rejection, so Qwen as the unattended default turns a
        # minutes-long stage into an hours-long one. A default should be the
        # choice a person would make without thinking; this is that choice.
        use_z_image_generation = z_image_turbo_models_ready and run.concept_backend in {
            "auto",
            "z_image_turbo",
        }
        use_qwen_image_generation = (
            qwen_image_2512_models_ready
            and run.concept_backend in {
                "auto",
                "qwen_image_2512",
                # Legacy persisted value: D1 now uses the proper generator model.
                "qwen_image_edit_2511",
            }
            and not use_z_image_generation
        )
        use_generator_model = use_qwen_image_generation or use_z_image_generation
        if not use_generator_model and run.checkpoint not in comfy.checkpoints():
            raise RuntimeError(f"required ComfyUI checkpoint is not installed: {run.checkpoint}")
        pose_guided = not use_generator_model and run.spec.anatomy_family == "humanoid" and (
            correction is None or correction.operation_id == "regenerate_complete_asset"
        )
        deferred_shield = _deferred_shield_equipment(run.spec) if pose_guided else None
        workflow_strategy = (
            "qwen_image_2512_t2i_v1"
            if use_qwen_image_generation
            else "z_image_turbo_t2i_v1"
            if use_z_image_generation
            else "pose_guided_lora_v3"
            if pose_guided
            else "local_crop_inpaint_v2"
            if correction is not None and correction.operation_id != "regenerate_complete_asset"
            else "global_text_v1"
        )
        control_guides: list[tuple[str, str, float, float, float]] = []
        guide_models: list[str] = []
        effective_positive, effective_negative = _whole_asset_prompt_guard(
            run.spec,
            plan.positive_prompt,
            plan.negative_prompt,
            defer_shield=not use_generator_model,
        )
        effective_positive, effective_negative = _geometry_ready_backdrop_guard(
            effective_positive,
            effective_negative,
        )
        applied_loras: list[tuple[str, float]] = [
            (name, strength)
            for name, strength in (
                (run.style_lora, run.style_lora_strength),
                (run.prop_lora, run.prop_lora_strength),
            )
            if not use_generator_model and name and name in installed_loras
        ]
        lora_trigger_words = ", ".join(
            trigger
            for name, trigger in ((run.style_lora, run.style_lora_trigger),)
            if not use_generator_model and trigger and name and name in installed_loras
        )
        if lora_trigger_words:
            effective_positive = f"{lora_trigger_words}, {effective_positive}"
        plain_positive = f"{lora_trigger_words}, {plan.positive_prompt}" if lora_trigger_words else plan.positive_prompt
        plain_positive, plain_negative = _geometry_ready_backdrop_guard(
            plain_positive,
            plan.negative_prompt,
        )
        if pose_guided:
            pose_path = make_humanoid_openpose_guide(attempt_root / "layout_guides")
            pose_model = next(
                (item for item in installed_controlnets if "openpose" in item.lower()),
                None,
            )
            self.store.evidence(
                run,
                "D1",
                pose_path,
                evidence_id=f"d1-i{stage.iteration:02d}-openpose-guide",
                label="Deterministic OpenPose structural guide",
                media_type="image/png",
                metrics={
                    "iteration": stage.iteration,
                    "selectable": False,
                    "role": "structural_guide",
                    "workflow_strategy": workflow_strategy,
                    "controlnet_model": pose_model or "not-installed",
                },
            )
            if pose_model is None:
                raise RuntimeError(
                    "A humanoid concept generation requires an installed OpenPose ControlNet so the "
                    "figure stays one coherent body (e.g. controlnet_openpose_sdxl_xinsir.safetensors "
                    "in ComfyUI's models/controlnet folder). None of the installed ControlNets matched "
                    f"'openpose'. Installed: {installed_controlnets or 'none'}."
                )
            upload_folder = f"text2model_studio/{run.run_id}/i{stage.iteration:02d}"
            uploaded = comfy.upload_image(pose_path.name, pose_path.read_bytes(), upload_folder)
            control_guides.append((pose_model, uploaded, 0.85, 0.0, 0.85))
            guide_models.append(pose_model)
        qwen_edit_instruction = None
        qwen_instruction_error = ""
        qwen_edit_prompt = effective_positive
        qwen_edit_negative = " " if use_qwen_image_generation else effective_negative
        if use_qwen_image_generation:
            rewriter = getattr(qwen, "image_edit_instruction", None)
            if callable(rewriter):
                try:
                    qwen_edit_instruction = rewriter(
                        run.spec,
                        stage,
                        plan,
                        source_handling="new_text_to_image",
                    )
                    qwen_edit_prompt = qwen_edit_instruction.prompt
                    qwen_edit_negative = qwen_edit_instruction.negative_prompt
                    qwen_edit_prompt, qwen_edit_negative = _geometry_ready_backdrop_guard(
                        qwen_edit_prompt,
                        qwen_edit_negative,
                    )
                    instruction_path = attempt_root / "qwen_image_edit_instruction.json"
                    _write_json(instruction_path, qwen_edit_instruction.model_dump(mode="json"))
                    self.store.evidence(
                        run,
                        "D1",
                        instruction_path,
                        evidence_id=f"d1-i{stage.iteration:02d}-qwen-image-edit-instruction",
                        label="Qwen-rewritten native image-edit instruction",
                        media_type="application/json",
                        metrics={
                            "iteration": stage.iteration,
                            "selectable": False,
                            "workflow_strategy": workflow_strategy,
                            "source_handling": qwen_edit_instruction.source_handling,
                        },
                    )
                except Exception as exc:
                    qwen_instruction_error = f"{type(exc).__name__}: {exc}"
        profile_path = attempt_root / "workflow_profile.json"
        _write_json(
            profile_path,
            {
                "schema_version": 1,
                "strategy": workflow_strategy,
                "concept_backend": (
                    "qwen_image_2512" if use_qwen_image_generation
                    else "z_image_turbo" if use_z_image_generation
                    else "sdxl"
                ),
                "checkpoint": run.checkpoint,
                "qwen_image_models_ready": qwen_image_2512_models_ready,
                "qwen_diffusion_model": "qwen_image_2512_fp8_e4m3fn.safetensors" if use_qwen_image_generation else None,
                "qwen_text_encoder": "qwen_2.5_vl_7b_fp8_scaled.safetensors" if use_qwen_image_generation else None,
                "qwen_vae": "qwen_image_vae.safetensors" if use_qwen_image_generation else None,
                "qwen_prompt_rewritten": qwen_edit_instruction is not None,
                "qwen_prompt_rewrite_error": qwen_instruction_error or None,
                "z_image_turbo_models_ready": z_image_turbo_models_ready,
                "z_image_diffusion_model": "z_image_turbo_int8_convrot.safetensors" if use_z_image_generation else None,
                "z_image_text_encoder": "qwen_3_4b_fp8_mixed.safetensors" if use_z_image_generation else None,
                "z_image_vae": "ae.safetensors" if use_z_image_generation else None,
                "installed_controlnets": installed_controlnets,
                "applied_controlnets": guide_models,
                "installed_loras": installed_loras,
                "applied_loras": [name for name, _ in applied_loras],
                "pose_guided": pose_guided,
                "local_high_resolution_crop": workflow_strategy == "local_crop_inpaint_v2",
                "ordinary_checkpoint_inpaint_encoder": "VAEEncodeForInpaint",
                "whole_asset_prompt_guard": run.spec.asset_kind == "character",
                "lora_trigger_words": lora_trigger_words,
                "device_policy": run.device_policy,
                "candidate_budget": candidate_budget,
                "concept_width": concept_width,
                "concept_height": concept_height,
                "tiled_vae": run.vae_tiling,
                "effective_positive_prompt": (
                    qwen_edit_prompt if use_qwen_image_generation
                    else plain_positive if use_z_image_generation
                    else effective_positive if pose_guided
                    else plain_positive
                ),
                "effective_negative_prompt": (
                    qwen_edit_negative if use_qwen_image_generation
                    else plain_negative if use_z_image_generation
                    else effective_negative if pose_guided
                    else plain_negative
                ),
            },
        )
        self.store.evidence(
            run,
            "D1",
            profile_path,
            evidence_id=f"d1-i{stage.iteration:02d}-workflow-profile",
            label=f"ComfyUI workflow profile, iteration {stage.iteration}",
            media_type="application/json",
            metrics={
                "iteration": stage.iteration,
                "selectable": False,
                "role": "workflow_profile",
                "workflow_strategy": workflow_strategy,
            },
        )
        candidates: list[tuple[str, Path, dict[str, object]]] = []
        source_upload = None
        mask_upload = None
        correction_source: Path | None = None
        correction_mask: Path | None = None
        correction_crop_box: tuple[int, int, int, int] | None = None
        if correction and not use_generator_model and correction.operation_id != "regenerate_complete_asset":
            source_item = next(
                (item for item in stage.evidence if item.evidence_id == correction.base_evidence_id),
                None,
            )
            if source_item is None or not source_item.media_type.startswith("image/"):
                raise RuntimeError("Qwen correction selected a base that is not a visual candidate")
            correction_source = self.store.artifact_path(run.run_id, source_item.relative_path)
            crop, correction_mask, correction_crop_box = _prepare_inpaint_crop(
                correction_source,
                correction.edit_box_normalized,
                attempt_root / "local_repair",
            )
            upload_folder = f"text2model_studio/{run.run_id}/i{stage.iteration:02d}"
            source_upload = comfy.upload_image(crop.name, crop.read_bytes(), upload_folder)
            mask_upload = comfy.upload_image(
                correction_mask.name,
                correction_mask.read_bytes(),
                upload_folder,
            )
            for role, path, label in (
                ("repair_crop", crop, "High-resolution local correction crop"),
                ("correction_mask", correction_mask, "Feathered local correction mask"),
            ):
                self.store.evidence(
                    run,
                    "D1",
                    path,
                    evidence_id=f"d1-i{stage.iteration:02d}-{role}",
                    label=f"{label}, iteration {stage.iteration}",
                    media_type="image/png",
                    metrics={
                        "iteration": stage.iteration,
                        "selectable": False,
                        "operation_id": correction.operation_id,
                        "base_evidence_id": correction.base_evidence_id,
                        "workflow_strategy": workflow_strategy,
                    },
                )
        # Both the shield-repair "measure before act" guard and the equipment
        # conformance check below need a vision-capable qwen provider. Real
        # StudioQwen always has visual_presence; a test double that models an
        # older/narrower interface (most of this suite's FakeQwen instances)
        # may not, and must fall back to the pre-existing behaviour exactly
        # -- unconditional repair, no conformance report -- rather than raise
        # AttributeError deep inside a stage that a hundred other tests drive
        # through without caring about this specific mechanism.
        qwen_supports_vision_presence = callable(getattr(qwen, "visual_presence", None))
        for index, seed in enumerate(candidate_seeds, start=1):
            evidence_id = f"d1-i{stage.iteration:02d}-candidate-{index}"
            self._progress(
                run,
                "D1",
                0.15 + (index - 1) * (0.58 / max(1, len(candidate_seeds))),
                f"ComfyUI is rendering concept {index} of {len(candidate_seeds)} (seed {seed}).",
                phase="concept_generation",
                current=index,
                total=len(candidate_seeds),
                unit="candidates",
            )
            if use_qwen_image_generation:
                workflow = qwen_image_2512_workflow(
                    prompt=qwen_edit_prompt,
                    negative_prompt=qwen_edit_negative,
                    seed=seed,
                    prefix=f"Text2ModelStudio/{run.run_id}/concept/i{stage.iteration:02d}_{index}",
                    width=concept_width,
                    height=concept_height,
                    tiled_vae=run.vae_tiling,
                )
            elif use_z_image_generation:
                workflow = z_image_turbo_workflow(
                    prompt=plain_positive,
                    negative_prompt=plain_negative,
                    seed=seed,
                    prefix=f"Text2ModelStudio/{run.run_id}/concept/i{stage.iteration:02d}_{index}",
                    width=concept_width,
                    height=concept_height,
                    tiled_vae=run.vae_tiling,
                )
            elif (
                correction
                and correction.operation_id != "regenerate_complete_asset"
                and source_upload
                and mask_upload
            ):
                workflow = inpaint_workflow(
                    checkpoint=run.checkpoint,
                    positive=plain_positive,
                    negative=plain_negative,
                    seed=seed,
                    prefix=f"Text2ModelStudio/{run.run_id}/concept/i{stage.iteration:02d}_{index}",
                    source_image=source_upload,
                    mask_image=mask_upload,
                    denoise=correction.denoise,
                    loras=applied_loras,
                    tiled_vae=run.vae_tiling,
                )
            else:
                workflow = concept_workflow(
                    checkpoint=run.checkpoint,
                    positive=effective_positive if pose_guided else plain_positive,
                    negative=effective_negative if pose_guided else plain_negative,
                    seed=seed,
                    prefix=f"Text2ModelStudio/{run.run_id}/concept/i{stage.iteration:02d}_{index}",
                    loras=applied_loras,
                    control_guides=control_guides,
                    steps=run.concept_steps,
                    cfg=run.concept_cfg,
                    width=concept_width,
                    height=concept_height,
                    tiled_vae=run.vae_tiling,
                )
            outputs = comfy.generate(
                workflow=workflow,
                destination=attempt_root / f"candidate-{index}",
            )
            image = outputs[0]
            if correction_source and correction_mask and correction_crop_box:
                image = _composite_inpaint_crop(
                    correction_source,
                    image,
                    correction_mask,
                    correction_crop_box,
                    attempt_root / f"candidate-{index}" / "composited_full_image.png",
                )
            shield_repaired = False
            shield_repair_skipped_reason = ""
            shield_duplicate_risk_similarity: float | None = None
            if not use_generator_model and deferred_shield is not None and not (
                correction and correction.operation_id != "regenerate_complete_asset"
            ):
                shield_dir = attempt_root / f"candidate-{index}" / "shield_repair"
                shield_box = _limb_repair_box(deferred_shield.side)
                # Measure before acting. This repair used to run
                # unconditionally, regardless of whether the base render had
                # already honored -- or, as observed on a real run, ignored --
                # its own "this side is empty" instruction. Only the target
                # region is checked here (a cheap, narrow question); a stray
                # item that landed somewhere ELSE on the character is not
                # this candidate's target box and is what the deterministic
                # check below and check_equipment_conformance() after the
                # loop exist to catch instead.
                # Short and generic (category noun, not the full spec
                # description) -- see spec_conformance.py's presence-check
                # comment for the live-verified reason: the verbose,
                # spec-worded phrasing measurably degrades the 4B reviewer's
                # accuracy on the exact same crop.
                already_present = qwen_supports_vision_presence and presence_check(
                    qwen,
                    image,
                    shield_box,
                    f"Does this image already clearly show a {deferred_shield.category}?",
                )
                if already_present:
                    shield_repair_skipped_reason = "already_present_in_target_region"
                else:
                    shield_crop, shield_mask, shield_crop_box = _prepare_inpaint_crop(
                        image, shield_box, shield_dir
                    )
                    shield_upload_folder = f"text2model_studio/{run.run_id}/i{stage.iteration:02d}"
                    shield_crop_upload = comfy.upload_image(
                        shield_crop.name, shield_crop.read_bytes(), shield_upload_folder
                    )
                    shield_mask_upload = comfy.upload_image(
                        shield_mask.name, shield_mask.read_bytes(), shield_upload_folder
                    )
                    shield_positive, shield_negative = _deferred_shield_repair_prompts(deferred_shield)
                    shield_workflow = inpaint_workflow(
                        checkpoint=run.checkpoint,
                        positive=shield_positive,
                        negative=shield_negative,
                        seed=seed + 1,
                        prefix=f"Text2ModelStudio/{run.run_id}/concept/i{stage.iteration:02d}_{index}_shield",
                        source_image=shield_crop_upload,
                        mask_image=shield_mask_upload,
                        denoise=0.75,
                        loras=applied_loras,
                        tiled_vae=run.vae_tiling,
                    )
                    shield_generated = comfy.generate(
                        workflow=shield_workflow,
                        destination=shield_dir / "generated",
                    )[0]
                    # Deterministic, model-free second layer, run before the
                    # paste: does the patch about to be pasted already closely
                    # resemble something present elsewhere in the base image?
                    # A near-duplicate here is the pixel-level signature of the
                    # observed bug (a stray shield the base render produced,
                    # plus a second one about to be pasted in). Recorded, not
                    # blocking on its own -- a stylized render can legitimately
                    # false-positive here -- so check_equipment_conformance()
                    # after the loop is the semantic, authoritative check.
                    crop_width = shield_crop_box[2] - shield_crop_box[0]
                    crop_height = shield_crop_box[3] - shield_crop_box[1]
                    _duplicate_found, shield_duplicate_risk_similarity = patch_already_present_elsewhere(
                        image,
                        shield_generated,
                        exclude_box=shield_crop_box,
                        template_resize=(max(1, crop_width), max(1, crop_height)),
                    )
                    image = _composite_inpaint_crop(
                        image, shield_generated, shield_mask, shield_crop_box, shield_dir / "composited.png"
                    )
                    shield_repaired = True
            equipment_conformance = (
                check_equipment_conformance(qwen, image, run.spec)
                if qwen_supports_vision_presence
                else EquipmentConformanceReport(conforms=True)
            )
            if equipment_conformance.violations:
                conformance_path = attempt_root / f"candidate-{index}" / "equipment_conformance.json"
                _write_json(conformance_path, equipment_conformance.model_dump(mode="json"))
                self.store.evidence(
                    run,
                    "D1",
                    conformance_path,
                    evidence_id=f"{evidence_id}-equipment-conformance",
                    label=f"Equipment contract check, candidate {index}",
                    media_type="application/json",
                    metrics={"iteration": stage.iteration, "selectable": False},
                )
            alpha_preview = attempt_root / f"candidate-{index}" / "geometry_ready_rgba.png"
            alpha_error = ""
            try:
                alpha_metrics = make_chroma_alpha(image, alpha_preview)
            except Exception as exc:
                alpha_metrics = {"meaningful_alpha": False}
                alpha_error = f"{type(exc).__name__}: {exc}"
                alpha_preview = None
            quality = assess_concept_image(
                image,
                alpha_preview,
                minimum_score=run.concept_min_quality_score,
            )
            if alpha_preview is not None:
                self.store.evidence(
                    run,
                    "D1",
                    alpha_preview,
                    evidence_id=f"{evidence_id}-geometry-ready-alpha",
                    label=f"Deterministic geometry-ready alpha, candidate {index}",
                    media_type="image/png",
                    metrics={
                        **alpha_metrics,
                        "iteration": stage.iteration,
                        "selectable": False,
                        "role": "geometry_ready_alpha",
                        "source_evidence_id": evidence_id,
                    },
                )
            quality_path = attempt_root / f"candidate-{index}" / "concept_quality.json"
            _write_json(
                quality_path,
                {
                    **quality.model_dump(mode="json"),
                    "alpha_error": alpha_error or None,
                    "equipment_conforms": equipment_conformance.conforms,
                    "equipment_violations": equipment_conformance.violations,
                },
            )
            self.store.evidence(
                run,
                "D1",
                quality_path,
                evidence_id=f"{evidence_id}-quality",
                label=f"Deterministic concept quality assessment, candidate {index}",
                media_type="application/json",
                metrics={
                    "iteration": stage.iteration,
                    "selectable": False,
                    "role": "concept_quality",
                    "quality_score": quality.score,
                    "quality_gate_passed": quality.hard_requirements_satisfied,
                },
            )
            metrics = _image_metrics(image)
            metrics.update(
                {
                    "seed": seed,
                    "iteration": stage.iteration,
                    # A candidate with a detected equipment duplicate must
                    # not be approvable at all -- StudioStore.decide() already
                    # hard-enforces this boolean at approval time, so this is
                    # the actual gate, not just a recorded finding. Without
                    # it, D1's own free-form critic (proven unreliable for
                    # counting -- see spec_conformance.py's module docstring)
                    # remains the only thing standing between a duplicated
                    # candidate and approval.
                    "selectable": (
                        equipment_conformance.conforms and quality.hard_requirements_satisfied
                    ),
                    "operation_id": correction.operation_id if correction else "regenerate_complete_asset",
                    "base_evidence_id": correction.base_evidence_id if correction else "",
                    "workflow_strategy": workflow_strategy,
                    "concept_backend": (
                        "qwen_image_2512" if use_qwen_image_generation
                        else "z_image_turbo" if use_z_image_generation
                        else "sdxl"
                    ),
                    "controlnet_count": len(control_guides),
                    "lora_count": len(applied_loras),
                    "local_high_resolution_crop": correction_crop_box is not None,
                    "deferred_shield_repaired": shield_repaired,
                    "deferred_shield_repair_skipped_reason": shield_repair_skipped_reason,
                    "shield_repair_duplicate_risk_similarity": shield_duplicate_risk_similarity,
                    "equipment_conformance_ok": equipment_conformance.conforms,
                    "equipment_conformance_violations": "; ".join(equipment_conformance.violations),
                    **quality.metrics,
                    "quality_reasons": "; ".join(quality.reasons),
                    "alpha_error": alpha_error,
                }
            )
            self.store.evidence(
                run,
                "D1",
                image,
                evidence_id=evidence_id,
                label=f"Concept {index}, iteration {stage.iteration}",
                media_type="image/png",
                metrics=metrics,
            )
            candidates.append((evidence_id, image, metrics))
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                item[2].get("selectable") is True,
                float(item[2].get("quality_score", 0.0)),
            ),
            reverse=True,
        )
        board_candidates = ranked_candidates[: run.concept_review_limit]
        review_candidates = [
            item for item in ranked_candidates if item[2].get("selectable") is True
        ][: run.concept_review_limit]
        comparison_board = _concept_comparison_board(
            self.store,
            run,
            stage,
            board_candidates,
            attempt_root / "previous_vs_current.png",
        )
        self.store.evidence(
            run,
            "D1",
            comparison_board,
            evidence_id=f"d1-i{stage.iteration:02d}-comparison-board",
            label=f"Previous rejection versus current candidates, iteration {stage.iteration}",
            media_type="image/png",
            metrics={"iteration": stage.iteration, "selectable": False},
        )
        self._progress(
            run,
            "D1",
            0.82,
            "Qwen critic is reading the deterministically pre-ranked evidence board.",
            phase="semantic_review",
            current=len(review_candidates),
            total=len(candidates),
            unit="ranked candidates",
        )
        if not review_candidates:
            review = StudioQwenReview(
                review_id=f"d1.deterministic-reject-{stage.iteration:02d}",
                stage_id="D1",
                iteration=stage.iteration,
                summary=(
                    "No concept reached semantic review because every candidate failed a deterministic "
                    "production gate. The images remain visible as non-approvable evidence."
                ),
                issues=["all candidates failed deterministic production gates"],
                candidate_ranking=[],
                recommended_evidence_id=None,
                recommended_changes=["Correct the recorded alpha, framing, or equipment failures and retry."],
                confidence=1.0,
                hard_requirements_satisfied=False,
                request_human_review=stage.iteration >= 6,
            )
        else:
            try:
                review = qwen.review_concepts(
                    run.spec,
                    stage,
                    review_candidates,
                    comparison_board=comparison_board,
                )
            except Exception as exc:
                review = StudioQwenReview(
                    review_id=f"d1.qwen.unavailable-{stage.iteration:02d}",
                    stage_id="D1",
                    iteration=stage.iteration,
                    summary=(
                        "The Qwen critic did not finish inside the bounded review window. The generated images are "
                        "available now; no candidate is automatically recommended."
                    ),
                    issues=[f"Qwen critic error: {type(exc).__name__}: {exc}"],
                    candidate_ranking=[item[0] for item in review_candidates],
                    recommended_evidence_id=None,
                    recommended_changes=["Use the human decision as the next optimizer instruction."],
                    confidence=0.0,
                    hard_requirements_satisfied=False,
                    request_human_review=True,
                )
        stage.qwen_reviews.append(review)
        review_path = attempt_root / "qwen_review.json"
        _write_json(review_path, review.model_dump(mode="json"))
        self.store.evidence(
            run,
            "D1",
            review_path,
            evidence_id=f"d1-i{stage.iteration:02d}-qwen-review",
            label=f"Qwen comparison, iteration {stage.iteration}",
            media_type="application/json",
            metrics={"confidence": review.confidence, "selectable": False},
        )
        stage.metrics = {
            "iteration": stage.iteration,
            "candidate_count": len(candidates),
            "selectable_candidate_count": sum(
                item[2].get("selectable") is True for item in candidates
            ),
            "best_deterministic_quality_score": max(
                (float(item[2].get("quality_score", 0.0)) for item in candidates),
                default=0.0,
            ),
            "review_candidate_count": len(review_candidates),
            "qwen_confidence": review.confidence,
            "recommended_evidence_id": review.recommended_evidence_id or "",
            "hard_requirements_satisfied": review.hard_requirements_satisfied,
            "workflow_strategy": workflow_strategy,
            "controlnet_count": len(control_guides),
            "lora_count": len(applied_loras),
        }
        stage.progress = 1
        stage.finished_at = utc_now()
        automatically_retry = (
            not any(item[2].get("selectable") is True for item in candidates)
            or (
                is_rejected_retry
                and not review.hard_requirements_satisfied
                and review.confidence >= 0.6
            )
        ) and stage.iteration < 6
        if automatically_retry:
            stage.state = "rejected"
            stage.message = (
                "No candidate cleared the deterministic and semantic requirements. "
                "The next bounded candidate search will run automatically."
            )
            run.state = "running"
            self.store.event(
                run,
                "qwen_referee_rejected",
                {
                    "stage_id": "D1",
                    "iteration": stage.iteration,
                    "summary": review.summary,
                },
            )
            return
        stage.state = "awaiting_review"
        stage.message = (
            "Six bounded searches produced no approvable concept. Review the recorded failures, then edit, "
            "retry, or roll back."
            if not review_candidates
            else "Concept candidates and Qwen comparison are ready for your approval or rejection."
        )
        run.state = "awaiting_review"
        self.store.event(
            run,
            "human_gate_ready",
            {
                "stage_id": "D1",
                "iteration": stage.iteration,
                "recommended_evidence_id": review.recommended_evidence_id,
            },
        )

    def _selected_evidence_path(self, run: StudioRun, stage_id: str) -> Path:
        stage = run.stage(stage_id)
        approvals = [item for item in stage.human_decisions if item.decision == "approve"]
        if not approvals or not approvals[-1].selected_evidence_id:
            raise RuntimeError(f"{stage_id} has no approved selected evidence")
        evidence_id = approvals[-1].selected_evidence_id
        evidence = next((item for item in stage.evidence if item.evidence_id == evidence_id), None)
        if evidence is None:
            raise RuntimeError(f"selected {stage_id} evidence no longer exists")
        return self.store.artifact_path(run.run_id, evidence.relative_path)

    def _latest_evidence_path(
        self,
        run: StudioRun,
        stage_id: str,
        *,
        media_type: str | None = None,
        role: str | None = None,
    ) -> Path:
        for item in reversed(run.stage(stage_id).evidence):
            if media_type is not None and item.media_type != media_type:
                continue
            if role is not None and item.metrics.get("role") != role:
                continue
            return self.store.artifact_path(run.run_id, item.relative_path)
        raise RuntimeError(f"{stage_id} has no evidence matching media_type={media_type!r}, role={role!r}")

    def _stage_settings(
        self,
        run: StudioRun,
        stage_id: str,
    ) -> dict[str, Any]:
        """Resolved [stages.<id>] configuration for this run's profile.

        Resolved per call rather than cached on the run so a profile edit
        takes effect on the next attempt; the run pins only WHICH profile
        (run.profile), not a frozen copy of its values. Returns {} if the
        profile or section is missing, so a stage always falls back to its
        own documented default rather than failing on configuration."""
        try:
            resolved = resolve_settings(profile=run.profile)
        except FileNotFoundError:
            return {}
        section = resolved.values.get("stages", {}).get(stage_id)
        return dict(section) if isinstance(section, dict) else {}

    def _run_script(self, command: list[str], *, timeout: float):
        """Run one resources/adapters/ helper script. Goes through the injected
        script_runner when there is one, so a test can substitute the
        Blender/ComfyUI-dependent scripts without also faking the
        dependency-light ones it can legitimately run for real."""
        runner = self._script_runner or subprocess.run
        return runner(command, text=True, capture_output=True, timeout=timeout)

    def _workspace_relative(self, path: Path) -> str:
        """Workspace-relative blob_path for an artifact record.

        Every Studio-run file lives under the store's own workspace by
        construction, so that is the correct base. config.local.toml's
        workspace_root is NOT: `text2model_forge studio --workspace X` explicitly
        allows X to differ from it, and basing on the config value made
        Path.relative_to() raise an opaque
        "'...' is not in the subpath of '...'" mid-stage whenever they
        diverged. Falls back to the resolved absolute path rather than
        raising if a caller ever passes something genuinely outside.
        """
        resolved = path.resolve()
        base = self.store.workspace
        if base == resolved or base in resolved.parents:
            return resolved.relative_to(base).as_posix()
        return resolved.as_posix()

    def _artifact_for_path(self, run: StudioRun, path: Path, *, artifact_id: str) -> ArtifactRecord:
        resolved = path.resolve()
        digest = sha256_file(resolved)
        return ArtifactRecord(
            artifact_id=artifact_id,
            sha256=digest,
            size_bytes=resolved.stat().st_size,
            media_type=(
                "application/x-blender"
                if resolved.suffix.lower() == ".blend"
                else "model/gltf-binary"
            ),
            stage=AssetStage.geometry,
            blob_path=self._workspace_relative(resolved),
            created_at=datetime.now(timezone.utc),
            lineage=ArtifactLineage(
                artifact_id=artifact_id,
                artifact_sha256=digest,
                stage=AssetStage.geometry.value,
                source_license_ids=["user-authored", "project-generated"],
            ),
            metadata={"studio_run_id": run.run_id},
        )

    def _execute_worker(
        self,
        worker_id: str,
        request: ExternalWorkerRequest,
        *,
        timeout_seconds: float,
    ):
        if self._worker_executor is not None:
            return self._worker_executor(worker_id, request, timeout_seconds=timeout_seconds)
        config = load_local_config()
        if config is None:
            raise RuntimeError("Text2Model config.local.toml is required")
        binding = worker_binding(config, worker_id)
        if binding is None:
            raise RuntimeError(f"Text2Model worker has no machine binding: {worker_id}")
        manifest = load_manifests()[worker_id]
        workspace = Path(config.workspace_root).resolve()
        adapter = SubprocessWorkerAdapter(
            WorkerManager(workspace, allowed_roots=[workspace]),
            manifest,
            binding.command_prefix,
            environment=binding.environment,
        )
        return adapter.execute(request, timeout_seconds=timeout_seconds)

    def _run_d2(self, run: StudioRun) -> None:
        stage = run.stage("D2")
        if run.spec is None:
            raise RuntimeError("D2 requires the compiled specification")
        selected_concept = self._selected_evidence_path(run, "D1")
        attempt_overrides = self._begin(run, "D2", "Turning the approved concept image into 3D geometry.")
        attempt_root = self.store.run_root(run.run_id) / "D2_geometry" / f"iteration-{stage.iteration:02d}"
        settings = self._stage_settings(run, "D2")
        settings.update(attempt_overrides)
        comfy = self._comfy_factory(run)
        self._register_comfy(run.run_id, comfy)
        # The image the human approved at D1 IS the image-to-3D input. It is
        # only background-keyed to RGBA first, because every image-to-3D model
        # needs the subject isolated from its background. D2 used to discard
        # the approved image and render a brand new one from a text prompt,
        # which meant "text -> 2D -> 3D" was never actually a chain: the mesh
        # came from an image nobody had reviewed.
        self._progress(run, "D2", 0.08, "Keying the approved concept to RGBA for image-to-3D.")
        rgba_seed = attempt_root / "geometry_seed_rgba.png"
        alpha_metrics = make_chroma_alpha(selected_concept, rgba_seed)
        alpha_metrics.update(
            {
                "iteration": stage.iteration,
                "source_evidence": "d1_approved_concept",
                "source_sha256": sha256_file(selected_concept),
            }
        )
        self.store.evidence(
            run,
            "D2",
            rgba_seed,
            evidence_id=f"d2-i{stage.iteration:02d}-rgba-seed",
            label=f"Geometry seed with deterministic alpha, iteration {stage.iteration}",
            media_type="image/png",
            metrics=alpha_metrics,
        )
        backend = str(settings.get("backend", "hunyuan3d"))
        configured_seed = int(settings.get("seed", 7132026))
        geometry_seed = configured_seed + max(0, stage.iteration - 1) * 104729
        if backend == "hunyuan3d":
            # Native ComfyUI Hunyuan3D-2: ~5GB (mini) to 6GB (standard) for
            # shape, versus 16-24GB for TRELLIS.2-4B. This is what makes the
            # image-to-3D step reachable on a consumer laptop GPU at all.
            self._progress(
                run,
                "D2",
                0.2,
                "Hunyuan3D is generating the mesh from the approved concept.",
                phase="geometry_generation",
                current=0,
                total=int(settings.get("steps", 20)),
                unit="scheduled diffusion steps",
            )
            uploaded = comfy.upload_image(
                f"{run.run_id}_d2_i{stage.iteration:02d}_rgba.png",
                rgba_seed.read_bytes(),
            )
            workflow = hunyuan3d_workflow(
                image=uploaded,
                prefix=f"Text2ModelStudio/{run.run_id}/geometry/i{stage.iteration:02d}",
                seed=geometry_seed,
                checkpoint=str(settings.get("checkpoint", "hunyuan3d-dit-v2-mini.safetensors")),
                steps=int(settings.get("steps", 50)),
                cfg=float(settings.get("cfg", 5.0)),
                octree_resolution=int(settings.get("octree_resolution", 256)),
            )
            produced = comfy.generate(
                workflow=workflow,
                destination=attempt_root / "hunyuan3d",
                timeout_seconds=float(settings.get("timeout_seconds", 1800)),
            )
            geometry_output = next(
                (item for item in produced if item.suffix.lower() in {".glb", ".gltf"}), None
            )
            if geometry_output is None:
                raise RuntimeError(
                    "Hunyuan3D produced no GLB. Check that ComfyUI has the Hunyuan3D checkpoint "
                    f"installed in models/checkpoints/ (got: {[p.name for p in produced]})"
                )
            worker_diagnostics: dict[str, object] = {
                "backend": "hunyuan3d",
                "octree_resolution": int(settings.get("octree_resolution", 256)),
            }
        else:
            # Typed subprocess worker (TRELLIS.2, TripoSG, the procedural
            # fallback). Kept for machines with the VRAM for it.
            artifact_id = f"{run.spec.asset_id}.d2.seed.i{stage.iteration:02d}"
            digest = sha256_file(rgba_seed)
            artifact = ArtifactRecord(
                artifact_id=artifact_id,
                sha256=digest,
                size_bytes=rgba_seed.stat().st_size,
                media_type="image/png",
                stage=AssetStage.concept,
                blob_path=self._workspace_relative(rgba_seed),
                created_at=datetime.now(timezone.utc),
                lineage=ArtifactLineage(
                    artifact_id=artifact_id,
                    artifact_sha256=digest,
                    stage=AssetStage.concept.value,
                    source_license_ids=["user-authored", "project-generated"],
                ),
                metadata={"meaningful_alpha": True, "studio_run_id": run.run_id},
            )
            self._progress(run, "D2", 0.2, f"{backend} is generating the 3D candidate from the approved RGBA input.")
            request = ExternalWorkerRequest(
                job_id=f"studio.{run.run_id}.d2.i{stage.iteration:02d}",
                run_id=run.run_id,
                operation_id="geometry.generate_from_rgba",
                stage=AssetStage.geometry,
                inputs=[artifact],
                input_paths={artifact_id: str(rgba_seed)},
                parameters={
                    "seed": geometry_seed,
                    "decimation_target": settings.get("decimation_target", 300000),
                    "texture_size": settings.get("texture_size", 2048),
                    "remesh": True,
                },
                output_directory=str(attempt_root / backend),
                device_policy=run.device_policy,
                gpu_safety_margin_gb=run.gpu_safety_margin_gb,
            )
            response = self._execute_worker(
                str(settings.get("worker_id", backend)),
                request,
                timeout_seconds=settings.get("timeout_seconds", 1800),
            )
            geometry_output = next(
                (Path(item.path) for item in response.outputs if item.media_type == "model/gltf-binary"),
                None,
            )
            if geometry_output is None:
                raise RuntimeError(f"{backend} returned no GLB candidate")
            worker_diagnostics = dict(response.diagnostics)
        self._progress(run, "D2", 0.78, "Building deterministic four-view geometry evidence.")
        diagnostic = attempt_root / "geometry_diagnostic.png"
        script = resource_root() / "adapters" / "render_glb_diagnostic.py"
        completed = self._run_script(
            [sys.executable, str(script), "--input", str(geometry_output), "--output", str(diagnostic)],
            timeout=300,
        )
        if completed.returncode != 0 or not diagnostic.is_file():
            raise RuntimeError(f"geometry diagnostic failed: {completed.stderr[-2000:]}")
        import trimesh

        loaded = trimesh.load(geometry_output, force="scene")
        meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("geometry output contains no mesh primitives")
        mesh = trimesh.util.concatenate(meshes)
        import numpy as np

        components = mesh.split(only_watertight=False)
        component_faces = [len(item.faces) for item in components]
        largest_component_fraction = max(component_faces, default=0) / max(1, len(mesh.faces))
        degenerate_face_fraction = float(np.count_nonzero(mesh.area_faces <= 1e-12)) / max(
            1, len(mesh.faces)
        )
        finite_vertices = bool(np.isfinite(mesh.vertices).all())
        nonzero_extents = bool(np.all(np.asarray(mesh.extents) > 1e-8))
        # Connectedness and degenerate faces are evidence here, not D2 hard
        # failures. Hunyuan/TRELLIS voxel extraction can be visibly coherent
        # while topologically unwelded; D3 owns welding/remeshing and its own
        # post-repair topology gates. D2 only proves that a substantive,
        # finite 3D candidate exists for that cleanup stage.
        hard_gate_passed = bool(
            len(mesh.vertices) >= 1000
            and len(mesh.faces) >= 1000
            and finite_vertices
            and nonzero_extents
        )
        metrics: dict[str, object] = {
            **worker_diagnostics,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "scene_meshes": len(meshes),
            "bounds_x": float(mesh.extents[0]),
            "bounds_y": float(mesh.extents[1]),
            "bounds_z": float(mesh.extents[2]),
            "connected_components": len(components),
            "largest_component_fraction": round(largest_component_fraction, 6),
            "degenerate_face_fraction": round(degenerate_face_fraction, 6),
            "finite_vertices": finite_vertices,
            "nonzero_extents": nonzero_extents,
            "hard_gate_passed": hard_gate_passed,
            "seed": geometry_seed,
        }
        self.store.evidence(
            run,
            "D2",
            geometry_output,
            evidence_id=f"d2-i{stage.iteration:02d}-glb",
            label=f"Geometry candidate from the approved concept, iteration {stage.iteration}",
            media_type="model/gltf-binary",
            metrics=metrics,
        )
        diagnostic_evidence = self.store.evidence(
            run,
            "D2",
            diagnostic,
            evidence_id=f"d2-i{stage.iteration:02d}-diagnostic",
            label=f"Four-view geometry diagnostic, iteration {stage.iteration}",
            media_type="image/png",
            metrics={**metrics, "iteration": stage.iteration},
        )
        self._progress(run, "D2", 0.88, "Qwen is comparing the mesh against the approved identity and numeric gates.")
        qwen = self._qwen_factory(run)
        review, qwen_passed = qwen.review_geometry(
            run.spec,
            stage,
            selected_concept,
            diagnostic,
            metrics,
        )
        stage.qwen_reviews.append(review)
        _write_json(attempt_root / "qwen_geometry_review.json", review.model_dump(mode="json"))
        hard_passed = bool(metrics["hard_gate_passed"])
        stage.metrics = {**metrics, "qwen_goal_satisfied": qwen_passed}
        stage.progress = 1
        stage.finished_at = utc_now()
        if hard_passed and qwen_passed:
            stage.state = "approved"
            stage.message = "Geometry passed hard gates and Qwen judged it credible for deterministic cleanup."
            self.store.event(run, "automatic_gate_passed", {"stage_id": "D2", "metrics": stage.metrics})
        elif stage.iteration < 3:
            stage.state = "rejected"
            stage.message = "Geometry did not clear the combined gate; Qwen will revise one bounded seed attempt."
            self.store.event(
                run,
                "automatic_gate_rejected",
                {"stage_id": "D2", "iteration": stage.iteration, "metrics": stage.metrics},
            )
        else:
            stage.gate_required = True
            stage.state = "awaiting_review"
            stage.message = "Three geometry attempts are preserved. Human direction is required before more GPU work."
            # Without this the gate is unactionable: StudioStore.decide()
            # refuses to approve evidence whose metrics do not say
            # selectable, and refuses to fall back to the recommendation for
            # the same reason -- so a human handed this gate could reject,
            # retry or roll back, but could never say "the automatic critic
            # was too strict, this mesh is fine". D3's identical escalation
            # already does this; D2's did not, which made a legitimate
            # override impossible exactly when the pipeline had given up.
            diagnostic_evidence.metrics["selectable"] = True
            run.state = "awaiting_review"
            review.recommended_evidence_id = diagnostic_evidence.evidence_id
            self.store.event(run, "human_gate_ready", {"stage_id": "D2", "iteration": stage.iteration})

    def _run_d3(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D3 requires the compiled asset specification")
        source = self._latest_evidence_path(run, "D2", media_type="model/gltf-binary")
        selected_concept = self._selected_evidence_path(run, "D1")
        attempt_overrides = self._begin(
            run, "D3", "Blender is validating and checkpointing deterministic geometry cleanup."
        )
        stage = run.stage("D3")
        attempt_root = self.store.run_root(run.run_id) / "D3_cleanup" / f"iteration-{stage.iteration:02d}"
        artifact_id = f"{run.spec.asset_id}.d3.input.i{stage.iteration:02d}"
        artifact = self._artifact_for_path(run, source, artifact_id=artifact_id)
        deformable = run.spec.behavior == "deformable_animated"
        maximum_components = 1 if deformable else max(1, len(run.spec.components) * 4, 32)
        settings = self._stage_settings(run, "D3")
        settings.update(attempt_overrides)
        request = ExternalWorkerRequest(
            job_id=f"studio.{run.run_id}.d3.i{stage.iteration:02d}",
            run_id=run.run_id,
            operation_id="blender.repair",
            stage=AssetStage.geometry,
            inputs=[artifact],
            input_paths={artifact_id: str(source)},
            parameters={
                "component_policy": "keep_largest" if deformable else "none",
                "weld_distance": (
                    settings.get("weld_distance", 0.00005) if deformable else 0.0
                ),
                "render_size": settings.get("render_size", 512),
                "maximum_material_change_fraction": settings.get(
                    "maximum_material_change_fraction", 0.03
                ),
                "minimum_connected_components": 1,
                "maximum_connected_components": maximum_components,
                # An isosurface backend (hunyuan3d on a small card) emits a
                # non-manifold surface that welding cannot repair; see
                # voxel_remesh_to_manifold() in resources/adapters/blender_worker.py.
                # "if_needed" leaves an already-clean mesh completely alone.
                "manifold_policy": settings.get("manifold_policy", "weld_only"),
                "voxel_fraction": settings.get("voxel_fraction", 0.006),
                "voxel_target_faces": settings.get("voxel_target_faces", 0),
                "maximum_material_change_fraction_remeshed": settings.get(
                    "maximum_material_change_fraction_remeshed", 0.25
                ),
            },
            output_directory=str(attempt_root / "blender"),
            device_policy=run.device_policy,
            gpu_safety_margin_gb=run.gpu_safety_margin_gb,
        )
        response = self._execute_worker(
            "blender", request, timeout_seconds=settings.get("timeout_seconds", 900)
        )
        image_records: list[tuple[str, Path]] = []
        for output in response.outputs:
            path = Path(output.path)
            safe_role = output.role.replace("_", "-")
            evidence_id = f"d3-i{stage.iteration:02d}-{safe_role}"
            metrics = {
                "iteration": stage.iteration,
                "selectable": False,
                "role": output.role,
            }
            self.store.evidence(
                run,
                "D3",
                path,
                evidence_id=evidence_id,
                label=f"Cleanup {output.role.replace('_', ' ')}",
                media_type=output.media_type,
                metrics=metrics,
            )
            if output.media_type.startswith("image/") and output.role.startswith("candidate_"):
                image_records.append((output.role, path))
        if not image_records:
            raise RuntimeError("Blender cleanup returned no candidate diagnostic views")
        diagnostic = _image_board(
            image_records,
            attempt_root / "cleanup_diagnostic.png",
            columns=min(4, len(image_records)),
        )
        diagnostic_evidence = self.store.evidence(
            run,
            "D3",
            diagnostic,
            evidence_id=f"d3-i{stage.iteration:02d}-diagnostic",
            label=f"Cleanup fixed-view diagnostic, iteration {stage.iteration}",
            media_type="image/png",
            metrics={
                **response.diagnostics,
                "iteration": stage.iteration,
                "selectable": False,
                "role": "cleanup_diagnostic",
            },
        )
        self._progress(run, "D3", 0.86, "Qwen is checking identity preservation against numeric cleanup gates.")
        qwen = self._qwen_factory(run)
        review, qwen_passed = qwen.review_cleanup(
            run.spec,
            stage,
            selected_concept,
            diagnostic,
            dict(response.diagnostics),
        )
        stage.qwen_reviews.append(review)
        _write_json(attempt_root / "qwen_cleanup_review.json", review.model_dump(mode="json"))
        stage.metrics = {
            **response.diagnostics,
            "qwen_goal_satisfied": qwen_passed,
            "maximum_connected_components": maximum_components,
        }
        stage.progress = 1
        stage.finished_at = utc_now()
        if qwen_passed:
            stage.state = "approved"
            stage.message = "Cleanup passed Blender export gates and Qwen identity review."
            self.store.event(run, "automatic_gate_passed", {"stage_id": "D3", "metrics": stage.metrics})
        elif stage.iteration < 3:
            stage.state = "rejected"
            stage.message = "Cleanup evidence failed Qwen identity review; one bounded retry will run."
            self.store.event(
                run,
                "automatic_gate_rejected",
                {"stage_id": "D3", "iteration": stage.iteration, "evidence": diagnostic_evidence.evidence_id},
            )
        else:
            stage.gate_required = True
            stage.state = "awaiting_review"
            stage.message = "Three cleanup attempts are preserved; human direction is required."
            diagnostic_evidence.metrics["selectable"] = True
            review.recommended_evidence_id = diagnostic_evidence.evidence_id
            run.state = "awaiting_review"
            self.store.event(run, "human_gate_ready", {"stage_id": "D3", "iteration": stage.iteration})

    def _run_d4(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D4 requires the compiled asset specification")
        source = self._latest_evidence_path(
            run,
            "D3",
            media_type="model/gltf-binary",
            role="candidate_geometry",
        )
        selected_concept = self._selected_evidence_path(run, "D1")
        stage_overrides = self._begin(run, "D4", "Building the typed canonical structure and its review evidence.")
        stage = run.stage("D4")
        attempt_root = self.store.run_root(run.run_id) / "D4_structure" / f"iteration-{stage.iteration:02d}"
        d4_settings = self._stage_settings(run, "D4")
        d4_settings.update(stage_overrides)
        qwen = self._qwen_factory(run)
        if run.spec.behavior == "rigid_articulated":
            diagnostic = self._latest_evidence_path(run, "D3", role="candidate_front")
            plan = qwen.rigid_structure_plan(run.spec, stage, selected_concept, diagnostic)
            plan_path = attempt_root / "rigid_structure_plan.json"
            _write_json(plan_path, plan.model_dump(mode="json"))
            overlay = _rigid_structure_overlay(
                diagnostic,
                plan,
                attempt_root / "rigid_structure_overlay.png",
            )
            self.store.evidence(
                run,
                "D4",
                plan_path,
                evidence_id=f"d4-i{stage.iteration:02d}-rigid-plan",
                label="Qwen rigid structure contract",
                media_type="application/json",
                metrics={"iteration": stage.iteration, "selectable": False, "role": "rigid_structure_plan"},
            )
            overlay_item = self.store.evidence(
                run,
                "D4",
                overlay,
                evidence_id=f"d4-i{stage.iteration:02d}-rigid-overlay",
                label="Rigid component boxes, pivots, axes, and limits",
                media_type="image/png",
                metrics={"iteration": stage.iteration, "selectable": True, "role": "rigid_structure_overlay"},
            )
            review = StudioQwenReview(
                review_id=f"d4.rigid-plan.iteration-{stage.iteration:02d}",
                stage_id="D4",
                iteration=stage.iteration,
                summary=(
                    f"Qwen proposed {len(plan.parts)} typed rigid movable parts. Human approval is required for "
                    "component segmentation, pivots, rotation axes, and joint limits before Blender separates geometry."
                ),
                candidate_ranking=[overlay_item.evidence_id],
                recommended_evidence_id=overlay_item.evidence_id,
                confidence=plan.confidence,
                hard_requirements_satisfied=True,
                request_human_review=True,
            )
            stage.qwen_reviews.append(review)
            stage.metrics = {"rigid_parts": len(plan.parts), "confidence": plan.confidence}
        else:
            artifact_id = f"{run.spec.asset_id}.d4.input.i{stage.iteration:02d}"
            artifact = self._artifact_for_path(run, source, artifact_id=artifact_id)
            # landmark_adjustments/weight_adjustments are a real per-attempt
            # correction channel -- resources/adapters/blender_worker.py applies them as
            # bounded landmark offsets and joint-pair weight transfers, and
            # rejects an out-of-shape value itself (see WEIGHT_JOINT_PAIRS).
            # A retry/edit decision is the only source for them: an ordinary
            # attempt must still send {}/[] so an unrelated stale correction
            # from a much earlier iteration can never resurface.
            request = ExternalWorkerRequest(
                job_id=f"studio.{run.run_id}.d4.i{stage.iteration:02d}",
                run_id=run.run_id,
                operation_id="blender.propose_short_biped_rig",
                stage=AssetStage.rig,
                inputs=[artifact],
                input_paths={artifact_id: str(source)},
                parameters={
                    "render_size": stage_overrides.get("render_size", d4_settings.get("render_size", 512)),
                    "maximum_material_change_fraction": stage_overrides.get(
                        "maximum_material_change_fraction",
                        d4_settings.get("maximum_material_change_fraction", 0.03),
                    ),
                    "maximum_bone_influences": stage_overrides.get(
                        "maximum_bone_influences", d4_settings.get("maximum_bone_influences", 4)
                    ),
                    "landmark_adjustments": stage_overrides.get("landmark_adjustments", {}),
                    "weight_adjustments": stage_overrides.get("weight_adjustments", []),
                },
                output_directory=str(attempt_root / "blender"),
                device_policy=run.device_policy,
                gpu_safety_margin_gb=run.gpu_safety_margin_gb,
            )
            response = self._execute_worker(
                "blender", request, timeout_seconds=d4_settings.get("timeout_seconds", 1200)
            )
            stress_records: list[tuple[str, Path]] = []
            for output in response.outputs:
                path = Path(output.path)
                evidence_id = f"d4-i{stage.iteration:02d}-{output.role.replace('_', '-')}"
                self.store.evidence(
                    run,
                    "D4",
                    path,
                    evidence_id=evidence_id,
                    label=f"Rig {output.role.replace('_', ' ')}",
                    media_type=output.media_type,
                    metrics={
                        "iteration": stage.iteration,
                        "selectable": False,
                        "role": output.role,
                    },
                )
                if output.media_type.startswith("image/") and output.role.startswith("rig_"):
                    stress_records.append((output.role, path))
            stress_board = _image_board(
                stress_records,
                attempt_root / "rig_stress_board.png",
                columns=4,
            )
            board_item = self.store.evidence(
                run,
                "D4",
                stress_board,
                evidence_id=f"d4-i{stage.iteration:02d}-rig-stress-board",
                label="Neutral and critical-joint deformation board",
                media_type="image/png",
                metrics={
                    **response.diagnostics,
                    "iteration": stage.iteration,
                    "selectable": True,
                    "role": "rig_stress_board",
                },
            )
            review = qwen.review_deformable_rig(
                run.spec,
                stage,
                selected_concept,
                stress_board,
                dict(response.diagnostics),
            )
            review.recommended_evidence_id = board_item.evidence_id
            stage.qwen_reviews.append(review)
            stage.metrics = dict(response.diagnostics)
        stage.progress = 1
        stage.state = "awaiting_review"
        stage.finished_at = utc_now()
        stage.message = "Canonical structure evidence is ready for your approval or rejection."
        run.state = "awaiting_review"
        self.store.event(run, "human_gate_ready", {"stage_id": "D4", "iteration": stage.iteration})

    def _adopt_d4_output(self, run: StudioRun, stage_id: str, roles: tuple[str, ...]) -> None:
        stage = run.stage(stage_id)
        self._begin(run, stage_id, f"Adopting hash-bound D4 outputs for {stage.label} validation.")
        adopted = 0
        gate_passed = True
        for item in run.stage("D4").evidence:
            role = str(item.metrics.get("role", ""))
            if role not in roles:
                continue
            path = self.store.artifact_path(run.run_id, item.relative_path)
            self.store.evidence(
                run,
                stage_id,
                path,
                evidence_id=f"{stage_id.lower()}-{role.replace('_', '-')}",
                label=f"Adopted {role.replace('_', ' ')}",
                media_type=item.media_type,
                metrics={"selectable": False, "role": role, "source_sha256": item.sha256},
            )
            adopted += 1
            if item.media_type == "application/json":
                value = json.loads(path.read_text(encoding="utf-8"))
                gate_passed = gate_passed and bool(
                    value.get("gate_passed", value.get("automatic_gate_passed", True))
                )
        if not adopted:
            raise RuntimeError(f"D4 did not produce the required {stage_id} evidence roles: {roles}")
        stage.metrics = {"adopted_artifacts": adopted, "source_stage": "D4", "hard_gate_passed": gate_passed}
        stage.progress = 1
        stage.finished_at = utc_now()
        if not gate_passed:
            raise RuntimeError(f"adopted {stage_id} report did not pass its hard gate")
        stage.state = "approved"
        stage.message = "Hash-bound canonical output passed deterministic adoption checks."
        self.store.event(run, "automatic_gate_passed", {"stage_id": stage_id, "metrics": stage.metrics})

    def _run_d5(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D5 requires the compiled asset specification")
        if run.spec.behavior == "rigid_articulated":
            self._begin(run, "D5", "Blender is separating approved rigid parts and authoring bounded pivots/actions.")
            stage = run.stage("D5")
            source = self._latest_evidence_path(
                run,
                "D3",
                media_type="application/x-blender",
                role="candidate_checkpoint",
            )
            plan_path = self._latest_evidence_path(run, "D4", role="rigid_structure_plan")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            attempt_root = self.store.run_root(run.run_id) / "D5_articulation" / f"iteration-{stage.iteration:02d}"
            artifact_id = f"{run.spec.asset_id}.d5.input.i{stage.iteration:02d}"
            artifact = self._artifact_for_path(run, source, artifact_id=artifact_id)
            request = ExternalWorkerRequest(
                job_id=f"studio.{run.run_id}.d5.i{stage.iteration:02d}",
                run_id=run.run_id,
                operation_id="blender.author_rigid_articulation",
                stage=AssetStage.rig,
                inputs=[artifact],
                input_paths={artifact_id: str(source)},
                parameters={
                    "render_size": 512,
                    "structure_plan": plan,
                    "animations": run.spec.animations,
                },
                output_directory=str(attempt_root / "blender"),
                device_policy=run.device_policy,
                gpu_safety_margin_gb=run.gpu_safety_margin_gb,
            )
            response = self._execute_worker("blender", request, timeout_seconds=1200)
            image_records: list[tuple[str, Path]] = []
            for output in response.outputs:
                path = Path(output.path)
                self.store.evidence(
                    run,
                    "D5",
                    path,
                    evidence_id=f"d5-i{stage.iteration:02d}-{output.role.replace('_', '-')}",
                    label=f"Rigid articulation {output.role.replace('_', ' ')}",
                    media_type=output.media_type,
                    metrics={"iteration": stage.iteration, "selectable": False, "role": output.role},
                )
                if output.media_type.startswith("image/"):
                    image_records.append((output.role, path))
            board = _image_board(image_records, attempt_root / "rigid_articulation_board.png", columns=4)
            self.store.evidence(
                run,
                "D5",
                board,
                evidence_id=f"d5-i{stage.iteration:02d}-articulation-board",
                label="Rigid neutral versus open articulation board",
                media_type="image/png",
                metrics={
                    **response.diagnostics,
                    "iteration": stage.iteration,
                    "selectable": False,
                    "role": "rigid_articulation_board",
                },
            )
            stage.metrics = dict(response.diagnostics)
            stage.progress = 1
            stage.state = "approved"
            stage.finished_at = utc_now()
            stage.message = "Rigid parts, pivots, degree limits, and requested actions passed deterministic gates."
            self.store.event(run, "automatic_gate_passed", {"stage_id": "D5", "metrics": stage.metrics})
            return
        self._adopt_d4_output(
            run,
            "D5",
            ("rig_contract", "landmarks_contract", "rigged_candidate_checkpoint", "rigged_candidate"),
        )

    def _run_d6(self, run: StudioRun) -> None:
        self._adopt_d4_output(
            run,
            "D6",
            ("skinning_report", "deformation_report", "neutral_comparison_report", "rigged_export_validation"),
        )

    # The clip D7 retargets when no catalog entry is selected. Kept as the
    # default so existing runs behave exactly as before the motion catalog
    # existed; set [stages.D7].donor_motion_id to choose another.
    _DEFAULT_DONOR_MOTION = (
        Path("sources")
        / "quaternius_universal_animation_library_standard"
        / "Universal Animation Library[Standard]"
        / "Unreal-Godot"
        / "UAL1_Standard.glb"
    )

    def _resolve_donor_motion(self, run: StudioRun, workspace_root: Path) -> Path:
        """The donor animation D7 retargets onto the rig.

        [stages.D7].donor_motion_id names a clip_id in the motion catalog
        (resources/motion_library/catalog.json, next to the installed package or in the
        working directory). Empty falls back to the historical hardcoded
        qualified CC0 clip, so this changed nothing for existing runs.
        """
        clip_id = str(self._stage_settings(run, "D7").get("donor_motion_id", "")).strip()
        if clip_id:
            catalog = _motion_catalog_path()
            if catalog is None:
                raise RuntimeError(
                    f"[stages.D7].donor_motion_id is {clip_id!r} but no resources/motion_library/catalog.json "
                    "was found; copy resources/motion_library/catalog.example.json and fill it in"
                )
            library = load_motion_library(catalog)
            # Raises with a clear message for an unknown id or a clip whose
            # file has not been downloaded yet -- never silently falls back,
            # which would retarget a motion the operator did not ask for.
            motion_source = resolve_donor_motion_path(
                library, clip_id, catalog_dir=catalog.parent
            ).resolve()
        else:
            motion_source = (workspace_root / self._DEFAULT_DONOR_MOTION).resolve()
        if not motion_source.is_file():
            raise RuntimeError(f"donor motion source is missing: {motion_source}")
        return motion_source

    def _run_motion_chain(self, run: StudioRun, *, stop_after: str, timeout_seconds: int = 3600) -> Path:
        if run.spec is None:
            raise RuntimeError("the motion chain requires an asset specification")
        config = load_local_config()
        if config is None:
            raise RuntimeError("Text2Model config.local.toml is required")
        binding = worker_binding(config, "blender")
        if binding is None or not binding.command_prefix:
            raise RuntimeError("the Blender worker binding is required")
        blender = Path(binding.command_prefix[0]).resolve()
        if not blender.is_file():
            raise RuntimeError(f"configured Blender executable does not exist: {blender}")
        target = self._latest_evidence_path(
            run,
            "D5",
            media_type="application/x-blender",
            role="rigged_candidate_checkpoint",
        )
        motion_source = self._resolve_donor_motion(run, Path(config.workspace_root))
        spec_path = self._latest_evidence_path(run, "D0", media_type="application/json")
        chain_root = self.store.run_root(run.run_id) / "D7_D10_chain"
        script = resource_root() / "adapters" / "run_motion_candidate_pipeline.py"
        command = [
            sys.executable,
            str(script),
            "--target",
            str(target),
            "--motion-source",
            str(motion_source),
            "--character-spec",
            str(spec_path),
            "--output-root",
            str(chain_root),
            "--blender",
            str(blender),
            "--repo-root",
            str(resource_root()),
            "--model",
            run.model,
            "--comfy-url",
            run.comfy_url,
            "--surface-checkpoint",
            run.checkpoint,
            "--timeout-seconds",
            "1200",
            "--stop-after",
            stop_after,
        ]
        completed = self._run_script(command, timeout=timeout_seconds)
        if completed.returncode != 0:
            raise RuntimeError(
                f"resumable D7-D10 chain failed at {stop_after}: "
                f"{(completed.stdout + completed.stderr)[-5000:]}"
            )
        return chain_root

    def _run_d7(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D7 requires the compiled asset specification")
        attempt_overrides = self._begin(
            run, "D7", "Building typed motion evidence and independent Qwen review."
        )
        stage = run.stage("D7")
        selected_concept = self._selected_evidence_path(run, "D1")
        if run.spec.behavior == "rigid_articulated":
            source = self._latest_evidence_path(run, "D5", role="rigid_articulation_board")
            item = self.store.evidence(
                run,
                "D7",
                source,
                evidence_id=f"d7-i{stage.iteration:02d}-rigid-motion-board",
                label="Rigid neutral versus actuated motion review",
                media_type="image/png",
                metrics={**run.stage("D5").metrics, "iteration": stage.iteration, "selectable": True},
            )
            qwen = self._qwen_factory(run)
            review = qwen.review_rigid_motion(
                run.spec,
                stage,
                selected_concept,
                source,
                dict(run.stage("D5").metrics),
            )
            review.recommended_evidence_id = item.evidence_id
            stage.qwen_reviews.append(review)
            stage.metrics = dict(run.stage("D5").metrics)
        else:
            self._progress(run, "D7", 0.08, "Retargeting qualified CC0 humanoid motions onto the Text2Model rig.")
            chain = self._run_motion_chain(
                run,
                stop_after="retarget_qwen_review",
                timeout_seconds={
                    **self._stage_settings(run, "D7"),
                    **attempt_overrides,
                }.get("timeout_seconds", 3600),
            )
            evidence_root = chain / "retarget" / "human_review"
            board = evidence_root / "all_motion_front_keyposes.png"
            mediator_path = evidence_root / "qwen_retarget_mediator.json"
            report_path = chain / "retarget" / "retarget_validation.json"
            if not board.is_file() or not mediator_path.is_file() or not report_path.is_file():
                raise RuntimeError("motion chain did not publish the required D7 evidence packet")
            board_item = self.store.evidence(
                run,
                "D7",
                board,
                evidence_id=f"d7-i{stage.iteration:02d}-motion-board",
                label="Idle, walk, attack, and death key-pose board",
                media_type="image/png",
                metrics={"iteration": stage.iteration, "selectable": True, "role": "motion_board"},
            )
            for name in ("attack_front_keyposes.png", "walk_front_keyposes.png", "death_front_keyposes.png"):
                path = evidence_root / name
                if path.is_file():
                    self.store.evidence(
                        run,
                        "D7",
                        path,
                        evidence_id=f"d7-i{stage.iteration:02d}-{path.stem.replace('_', '-')}",
                        label=path.stem.replace("_", " ").title(),
                        media_type="image/png",
                        metrics={"iteration": stage.iteration, "selectable": False},
                    )
            mediator = json.loads(mediator_path.read_text(encoding="utf-8"))
            passed = mediator.get("corrected_overall") == "ready_for_human_gate"
            review = StudioQwenReview(
                review_id=f"d7.motion-mediator.iteration-{stage.iteration:02d}",
                stage_id="D7",
                iteration=stage.iteration,
                summary=str(mediator.get("reason", "Motion mediator completed.")),
                strengths=[str(item) for item in mediator.get("supported_claims", [])],
                issues=[str(item) for item in mediator.get("unsupported_or_overstated_claims", [])],
                candidate_ranking=[board_item.evidence_id],
                recommended_evidence_id=board_item.evidence_id,
                recommended_changes=[],
                confidence=0.8 if passed else 0.45,
                hard_requirements_satisfied=passed,
                request_human_review=True,
            )
            stage.qwen_reviews.append(review)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            stage.metrics = {
                "actions": len(report.get("actions", [])),
                "automatic_gate_passed": report.get("automatic_gate_passed", True),
                "qwen_mediator_passed": passed,
            }
        stage.progress = 1
        stage.state = "awaiting_review"
        stage.finished_at = utc_now()
        stage.message = "Motion evidence is ready for your approval or rejection."
        run.state = "awaiting_review"
        self.store.event(run, "human_gate_ready", {"stage_id": "D7", "iteration": stage.iteration})

    def _configured_blender(self) -> Path:
        config = load_local_config()
        if config is None:
            raise RuntimeError("Text2Model config.local.toml is required")
        binding = worker_binding(config, "blender")
        if binding is None or not binding.command_prefix:
            raise RuntimeError("the Blender worker binding is required")
        executable = Path(binding.command_prefix[0]).resolve()
        if not executable.is_file():
            raise RuntimeError(f"configured Blender executable does not exist: {executable}")
        return executable

    def _run_d8(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D8 requires the compiled asset specification")
        self._begin(run, "D8", "Preparing semantic materials, painting canonical views, and baking one stable master.")
        stage = run.stage("D8")
        attempt_root = self.store.run_root(run.run_id) / "D8_surface" / f"iteration-{stage.iteration:02d}"
        if run.spec.behavior == "deformable_animated":
            master = self.store.run_root(run.run_id) / "D7_D10_chain" / "retarget" / "quaternius_retargeted_candidate.blend"
            surface = self.store.run_root(run.run_id) / "D7_D10_chain" / "surface"
        elif run.spec.behavior == "rigid_articulated":
            master = self._latest_evidence_path(
                run,
                "D5",
                media_type="application/x-blender",
                role="rigid_articulated_checkpoint",
            )
            surface = attempt_root / "production"
        else:
            master = self._latest_evidence_path(
                run,
                "D3",
                media_type="application/x-blender",
                role="candidate_checkpoint",
            )
            surface = attempt_root / "production"
        if not master.is_file():
            raise RuntimeError(f"D8 source master is missing: {master}")
        original_spec = self._latest_evidence_path(run, "D0", media_type="application/json")
        spec_value = json.loads(original_spec.read_text(encoding="utf-8"))
        latest_rejection = latest_correction(stage)
        if latest_rejection is not None:
            qwen = self._qwen_factory(run)
            revision = qwen.revision_plan(run.spec, stage)
            spec_value["creative_direction"] = (
                str(spec_value["creative_direction"])
                + " Human surface correction: "
                + latest_rejection.comment
                + " Qwen bounded changes: "
                + "; ".join(revision.changes)
            )
            spec_value["negative_constraints"] = list(spec_value.get("negative_constraints", [])) + [
                latest_rejection.comment
            ]
        effective_spec = attempt_root / "effective_asset_spec.json"
        _write_json(effective_spec, spec_value)
        blender = self._configured_blender()
        script = resource_root() / "adapters" / "bake_surface.py"
        command = [
            sys.executable,
            str(script),
            "--master",
            str(master),
            "--asset-spec",
            str(effective_spec),
            "--output-directory",
            str(surface),
            "--blender",
            str(blender),
            "--repo-root",
            str(resource_root()),
            "--comfy-url",
            run.comfy_url,
            "--checkpoint",
            run.checkpoint,
            "--timeout-seconds",
            "1200",
        ]
        if latest_rejection is not None:
            command.append("--force")
        completed = self._run_script(command, timeout=3600)
        if completed.returncode != 0:
            raise RuntimeError(f"D8 surface bake failed: {(completed.stdout + completed.stderr)[-5000:]}")
        review_script = resource_root() / "adapters" / "review_surface_master.py"
        reviewed = self._run_script(
            [sys.executable, str(review_script), "--surface-directory", str(surface), "--model", run.model],
            timeout=300,
        )
        if reviewed.returncode != 0:
            raise RuntimeError(f"D8 surface review failed: {(reviewed.stdout + reviewed.stderr)[-3000:]}")
        board = surface / "surface_review.png"
        master_output = surface / "text2model_surface_master.blend"
        report_path = surface / "surface_validation.json"
        mediator_path = surface / "qwen_surface_mediator.json"
        for required in (board, master_output, report_path, mediator_path):
            if not required.is_file():
                raise RuntimeError(f"D8 did not publish required evidence: {required}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mediator = json.loads(mediator_path.read_text(encoding="utf-8"))
        board_item = self.store.evidence(
            run,
            "D8",
            board,
            evidence_id=f"d8-i{stage.iteration:02d}-surface-board",
            label="Before versus persistent painted surface master",
            media_type="image/png",
            metrics={
                **report.get("image_metrics", {}),
                "iteration": stage.iteration,
                "selectable": True,
                "automatic_gate_passed": report.get("automatic_gate_passed", False),
            },
        )
        for evidence_id, label, path, media_type, role in (
            (f"d8-i{stage.iteration:02d}-master", "Persistent painted Blender master", master_output, "application/x-blender", "surface_master"),
            (f"d8-i{stage.iteration:02d}-validation", "Surface numeric and provenance validation", report_path, "application/json", "surface_validation"),
            (f"d8-i{stage.iteration:02d}-mediator", "Qwen surface mediator", mediator_path, "application/json", "surface_mediator"),
        ):
            self.store.evidence(
                run,
                "D8",
                path,
                evidence_id=evidence_id,
                label=label,
                media_type=media_type,
                metrics={"iteration": stage.iteration, "selectable": False, "role": role},
            )
        passed = bool(report.get("automatic_gate_passed")) and mediator.get("corrected_overall") in {
            "ready_for_final_render",
            "uncertain",
        }
        review = StudioQwenReview(
            review_id=f"d8.surface-mediator.iteration-{stage.iteration:02d}",
            stage_id="D8",
            iteration=stage.iteration,
            summary=str(mediator.get("reason", "Surface evidence is ready for human review.")),
            strengths=[],
            issues=[str(item) for item in mediator.get("unsupported_or_overstated_claims", [])],
            candidate_ranking=[board_item.evidence_id],
            recommended_evidence_id=board_item.evidence_id,
            recommended_changes=[],
            confidence=0.8 if passed else 0.4,
            hard_requirements_satisfied=bool(report.get("automatic_gate_passed")),
            request_human_review=True,
        )
        stage.qwen_reviews.append(review)
        stage.metrics = {
            "automatic_gate_passed": report.get("automatic_gate_passed", False),
            "surface_master_sha256": report.get("surface_master_sha256", ""),
            "qwen_mediator": mediator.get("corrected_overall", "uncertain"),
        }
        stage.progress = 1
        stage.state = "awaiting_review"
        stage.finished_at = utc_now()
        stage.message = "Persistent painted surface evidence is ready for your approval or rejection."
        run.state = "awaiting_review"
        self.store.event(run, "human_gate_ready", {"stage_id": "D8", "iteration": stage.iteration})

    def _run_d9(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D9 requires the compiled asset specification")
        attempt_overrides = self._begin(
            run, "D9", "Rendering deterministic delivery views and validating the package."
        )
        stage = run.stage("D9")
        if run.spec.behavior == "deformable_animated":
            self._progress(run, "D9", 0.08, "Rendering and packaging directional motion sprites from one surface master.")
            chain = self._run_motion_chain(run, stop_after="sprite_qwen_review")
            package = chain / "sprites" / "package"
            board = package / "sprite_review.png"
            manifest = package / "candidate_unit_manifest.json"
            mediator_path = package / "qwen_sprite_mediator.json"
            for required in (board, manifest, mediator_path):
                if not required.is_file():
                    raise RuntimeError(f"D9 sprite chain did not publish {required}")
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            mediator = json.loads(mediator_path.read_text(encoding="utf-8"))
            passed = bool(manifest_value.get("automatic_gate_passed")) and mediator.get(
                "corrected_overall"
            ) == "ready_for_unity_candidate"
            self.store.evidence(
                run,
                "D9",
                board,
                evidence_id=f"d9-i{stage.iteration:02d}-sprite-board",
                label="Directional motion sprite package",
                media_type="image/png",
                metrics={
                    "iteration": stage.iteration,
                    "selectable": False,
                    "role": "delivery_board",
                    "automatic_gate_passed": passed,
                },
            )
            for evidence_id, path, label, role in (
                (f"d9-i{stage.iteration:02d}-manifest", manifest, "Hash-bound sprite manifest", "delivery_manifest"),
                (f"d9-i{stage.iteration:02d}-mediator", mediator_path, "Qwen sprite mediator", "delivery_mediator"),
            ):
                self.store.evidence(
                    run,
                    "D9",
                    path,
                    evidence_id=evidence_id,
                    label=label,
                    media_type="application/json",
                    metrics={"iteration": stage.iteration, "selectable": False, "role": role},
                )
            stage.metrics = {
                "automatic_gate_passed": passed,
                "qwen_mediator": mediator.get("corrected_overall", "uncertain"),
                "source_master_sha256": manifest_value.get("source_master_sha256", ""),
            }
        else:
            source = self._latest_evidence_path(
                run,
                "D8",
                media_type="application/x-blender",
                role="surface_master",
            )
            attempt_root = self.store.run_root(run.run_id) / "D9_delivery" / f"iteration-{stage.iteration:02d}"
            d9_settings = self._stage_settings(run, "D9")
            d9_settings.update(attempt_overrides)
            artifact_id = f"{run.spec.asset_id}.d9.input.i{stage.iteration:02d}"
            artifact = self._artifact_for_path(run, source, artifact_id=artifact_id)
            request = ExternalWorkerRequest(
                job_id=f"studio.{run.run_id}.d9.i{stage.iteration:02d}",
                run_id=run.run_id,
                operation_id="blender.render_diagnostics",
                stage=AssetStage.optimization,
                inputs=[artifact],
                input_paths={artifact_id: str(source)},
                parameters={"render_size": d9_settings.get("render_size", 768)},
                output_directory=str(attempt_root / "blender"),
                device_policy=run.device_policy,
                gpu_safety_margin_gb=run.gpu_safety_margin_gb,
            )
            response = self._execute_worker(
                "blender", request, timeout_seconds=d9_settings.get("timeout_seconds", 900)
            )
            records = [
                (output.role, Path(output.path))
                for output in response.outputs
                if output.media_type.startswith("image/")
            ]
            board = _image_board(records, attempt_root / "delivery_board.png", columns=4)
            board_item = self.store.evidence(
                run,
                "D9",
                board,
                evidence_id=f"d9-i{stage.iteration:02d}-delivery-board",
                label="Painted multi-view delivery board",
                media_type="image/png",
                metrics={
                    **response.diagnostics,
                    "iteration": stage.iteration,
                    "selectable": False,
                    "role": "delivery_board",
                },
            )
            manifest_path = attempt_root / "delivery_manifest.json"
            _write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "asset_id": run.spec.asset_id,
                    "asset_kind": run.spec.asset_kind,
                    "behavior": run.spec.behavior,
                    "source_master": str(source),
                    "source_master_sha256": sha256_file(source),
                    "delivery_board": str(board),
                    "delivery_board_sha256": board_item.sha256,
                    "automatic_gate_passed": bool(records),
                    "human_approval_required": True,
                    "human_approved": False,
                },
            )
            self.store.evidence(
                run,
                "D9",
                manifest_path,
                evidence_id=f"d9-i{stage.iteration:02d}-manifest",
                label="Hash-bound generic delivery manifest",
                media_type="application/json",
                metrics={"iteration": stage.iteration, "selectable": False, "role": "delivery_manifest"},
            )
            stage.metrics = {**response.diagnostics, "automatic_gate_passed": bool(records)}
        stage.progress = 1
        stage.finished_at = utc_now()
        if stage.metrics.get("automatic_gate_passed") is not True:
            raise RuntimeError("D9 delivery package failed its automatic gate")
        stage.state = "approved"
        stage.message = "Delivery renders, hashes, framing, and package contract passed automatic gates."
        self.store.event(run, "automatic_gate_passed", {"stage_id": "D9", "metrics": stage.metrics})

    def _run_d10(self, run: StudioRun) -> None:
        if run.spec is None:
            raise RuntimeError("D10 requires the compiled asset specification")
        self._begin(run, "D10", "Building a standalone runtime-review package and final human evidence.")
        stage = run.stage("D10")
        delivery_board = self._latest_evidence_path(run, "D9", role="delivery_board")
        runtime_link: Path | None = None
        if run.spec.behavior == "deformable_animated":
            chain = self._run_motion_chain(run, stop_after="unity_smoke_bundle")
            runtime_link = chain / "unity_smoke_bundle" / "review.html"
            runtime_manifest = chain / "unity_smoke_bundle" / "bundle_manifest.json"
            if not runtime_link.is_file() or not runtime_manifest.is_file():
                raise RuntimeError("D10 standalone Unity/browser bundle is incomplete")
        else:
            output = self.store.run_root(run.run_id) / "D10_runtime"
            output.mkdir(parents=True, exist_ok=True)
            runtime_manifest = output / "runtime_manifest.json"
            delivery_manifest = self._latest_evidence_path(run, "D9", role="delivery_manifest")
            _write_json(
                runtime_manifest,
                {
                    "schema_version": 1,
                    "asset_id": run.spec.asset_id,
                    "asset_kind": run.spec.asset_kind,
                    "behavior": run.spec.behavior,
                    "delivery_manifest": str(delivery_manifest),
                    "delivery_manifest_sha256": sha256_file(delivery_manifest),
                    "runtime_profile": "standalone_generic_asset_review_v1",
                    "automatic_gate_passed": True,
                    "human_approval_required": True,
                    "human_approved": False,
                },
            )
        final_item = self.store.evidence(
            run,
            "D10",
            delivery_board,
            evidence_id=f"d10-i{stage.iteration:02d}-final-board",
            label="Final runtime-scale asset evidence",
            media_type="image/png",
            metrics={
                "iteration": stage.iteration,
                "selectable": True,
                "role": "final_runtime_board",
            },
        )
        self.store.evidence(
            run,
            "D10",
            runtime_manifest,
            evidence_id=f"d10-i{stage.iteration:02d}-runtime-manifest",
            label="Standalone runtime package manifest",
            media_type="application/json",
            metrics={"iteration": stage.iteration, "selectable": False, "role": "runtime_manifest"},
        )
        if runtime_link is not None:
            self.store.evidence(
                run,
                "D10",
                runtime_link,
                evidence_id=f"d10-i{stage.iteration:02d}-browser-review",
                label="Interactive standalone motion review",
                media_type="text/html",
                metrics={"iteration": stage.iteration, "selectable": False, "role": "browser_review"},
            )
        review = StudioQwenReview(
            review_id=f"d10.package.iteration-{stage.iteration:02d}",
            stage_id="D10",
            iteration=stage.iteration,
            summary=(
                "The hash-bound standalone package passed automatic assembly. Human ship approval remains required "
                "for identity, motion/readability when applicable, surface quality, and runtime-scale appearance."
            ),
            candidate_ranking=[final_item.evidence_id],
            recommended_evidence_id=final_item.evidence_id,
            confidence=0.75,
            hard_requirements_satisfied=True,
            request_human_review=True,
        )
        stage.qwen_reviews.append(review)
        stage.metrics = {"automatic_gate_passed": True, "runtime_manifest_sha256": sha256_file(runtime_manifest)}
        stage.progress = 1
        stage.state = "awaiting_review"
        stage.finished_at = utc_now()
        stage.message = "Final standalone package is ready for explicit human ship approval."
        run.state = "awaiting_review"
        self.store.event(run, "human_gate_ready", {"stage_id": "D10", "iteration": stage.iteration})
