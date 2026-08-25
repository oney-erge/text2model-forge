"""Check every cross-stage assumption before a run, not three stages into one.

The 8 GB bring-up failed ten times in a row, each failure costing a full
pipeline run, and every one of them was knowable in advance:

    D0  the reviewer is too large to fit beside ComfyUI and spills to CPU
    D1  the concept backdrop is not what D2's keyer requires
    D2  ComfyUI has no Hunyuan3D nodes; the checkpoint is not installed
    D3  Blender's render-engine enum does not contain BLENDER_EEVEE_NEXT
    D3  the voxel is coarser than the fingers D7 will look for
    D7  the donor motion file is absent
    D7  the donor's bones are named for Rigify, not Unreal

`text2model_forge workers` already answers "does each worker exist". None of the
above is about a worker existing; they are about whether one stage's output
satisfies the next stage's assumptions on *this* machine with *this*
configuration. That is what this module checks.

The point is not that these particular seven never recur -- a new backend
will always bring new assumptions. The point is the failure mode: finding
all of them in one thirty-second report instead of one per twenty-minute
run. Checks are therefore independent and run concurrently; one failure
never hides another.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .config import load_local_config, worker_binding
from .gpu import gpu_memory_snapshot
from .hardware import (
    FINGER_WIDTH_M,
    REVIEWER_VRAM_GB,
    HardwareProfile,
    detect_hardware,
    recommend_stack,
    reviewer_vram_requirement,
)
from .motion_library import load_motion_library, resolve_donor_motion_path
from .paths import resource_root
from .schemas import StrictModel
from .settings import resolve_settings, studio_overrides


class Check(StrictModel):
    name: str = Field(min_length=1)
    status: Literal["ok", "warn", "fail", "skip"]
    detail: str = ""
    # What to actually do about it. A check that reports a problem without a
    # remedy just moves the guessing somewhere else.
    remedy: str = ""


def _http_json(url: str, timeout: float = 8.0) -> Any | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


# ComfyUI node types each D2 backend needs. Checking the *running server's*
# object_info is the only honest test: the nodes ship with ComfyUI core, so
# "is ComfyUI installed" says nothing about whether it is new enough.
_D2_REQUIRED_NODES: dict[str, tuple[str, ...]] = {
    "hunyuan3d": (
        "ImageOnlyCheckpointLoader",
        "Hunyuan3Dv2Conditioning",
        "EmptyLatentHunyuan3Dv2",
        "VAEDecodeHunyuan3D",
        "SaveGLB",
    ),
}


def check_reviewer_fits(hardware: HardwareProfile, settings: dict[str, Any]) -> Check:
    model = str(settings.get("model", ""))
    total = hardware.vram_total_gb
    if total is None:
        return Check(
            name="reviewer fits in VRAM",
            status="skip",
            detail="no GPU detected",
            remedy="Install nvidia-smi or run LocalDeploy so the budget can be computed.",
        )
    recommendation = recommend_stack(hardware)
    budget = recommendation.vram_budget_for_reviewer_gb or 0.0
    size = next(
        (key for key in sorted(REVIEWER_VRAM_GB, key=len, reverse=True) if key in model.lower()),
        None,
    )
    if size is None:
        return Check(
            name="reviewer fits in VRAM",
            status="warn",
            detail=f"cannot infer parameter count from model id {model!r}",
            remedy=f"Budget beside ComfyUI is {budget:.1f} GB; a {recommendation.reviewer_size} "
            "model is the largest that fits.",
        )
    need = REVIEWER_VRAM_GB[size]
    if need <= budget:
        return Check(
            name="reviewer fits in VRAM",
            status="ok",
            detail=f"{model} needs ~{need:.1f} GB, {budget:.1f} GB available beside ComfyUI",
        )
    return Check(
        name="reviewer fits in VRAM",
        status="fail",
        detail=(
            f"{model} needs ~{need:.1f} GB but only {budget:.1f} GB is free beside ComfyUI "
            f"({total:.1f} GB card). It will spill to CPU; a vision tower on CPU is ~100x "
            "slower and every review will look like a timeout."
        ),
        remedy=f"Set [studio_defaults].model to a {recommendation.reviewer_size}-class vision "
        "model, or stop ComfyUI while the reviewer runs.",
    )


def check_device_policy(
    hardware: HardwareProfile,
    settings: dict[str, Any],
    stages: dict[str, Any],
) -> Check:
    policy = str(settings.get("device_policy", "prefer_gpu"))
    if policy == "prefer_gpu":
        return Check(
            name="GPU execution policy",
            status="warn",
            detail="prefer_gpu permits an adapter to use CPU inference",
            remedy='Set [studio_defaults].device_policy = "gpu_compute_only" to forbid silent CPU inference.',
        )
    if hardware.primary is None:
        return Check(
            name="GPU execution policy",
            status="fail",
            detail=f"device_policy={policy} but no supported GPU was detected",
            remedy="Install a supported accelerator driver and verify it with text2model-forge doctor.",
        )
    if policy == "strict_device_only":
        manifold = str((stages.get("D3") or {}).get("manifold_policy", "weld_only"))
        return Check(
            name="GPU execution policy",
            status="fail",
            detail=(
                "strict_device_only cannot execute the current Blender topology, rigging, packaging, and "
                f"runtime-import stages (D3 manifold_policy={manifold})"
            ),
            remedy='Use device_policy = "gpu_compute_only". It keeps all learned inference and rendering on GPU '
            "while allowing deterministic orchestration and mesh operations on CPU.",
        )
    return Check(
        name="GPU execution policy",
        status="ok",
        detail=(
            f"gpu_compute_only on {hardware.primary.name}: explicit CPU inference is rejected; "
            "deterministic CPU orchestration remains allowed"
        ),
    )


def check_gpu_safety_margin(hardware: HardwareProfile, settings: dict[str, Any]) -> Check:
    margin = float(settings.get("gpu_safety_margin_gb", 0.75))
    gpu = hardware.primary
    if gpu is None or gpu.vram_total_gb is None:
        return Check(
            name="GPU safety margin",
            status="skip",
            detail="VRAM total is unknown",
            remedy="Enable supported GPU telemetry before using fail-closed admission.",
        )
    if margin < 0.5:
        recommended = 0.75
        return Check(
            name="GPU safety margin",
            status="fail",
            detail=f"{margin:.2f} GiB leaves too little display/driver/allocator headroom",
            remedy=f"Set gpu_safety_margin_gb to at least {recommended:.2f}.",
        )
    return Check(
        name="GPU safety margin",
        status="ok",
        detail=f"{margin:.2f} GiB reserved from {gpu.vram_total_gb:.2f} GiB total",
    )


def check_spec_strategy(settings: dict[str, Any]) -> Check:
    model = str(settings.get("model", "")).lower()
    strategy = str(settings.get("spec_strategy", "monolithic"))
    small = any(token in model for token in ("0.5b", "1b", "2b", "3b", "4b", "7b", "8b", "12b", "13b"))
    if small and strategy != "chunked":
        return Check(
            name="spec strategy suits the model",
            status="fail",
            detail=f"{model} is a small model but spec_strategy={strategy!r}",
            remedy='Set [studio_defaults].spec_strategy = "chunked". A single-call spec compile '
            "returns `equipment: []` below ~27B and fails D0's handedness contract.",
        )
    return Check(
        name="spec strategy suits the model",
        status="ok",
        detail=f"{model or 'unset'} with spec_strategy={strategy}",
    )


# Name tokens for reviewer models actually capable of answering an
# image-grounded question. Inferred from the model id because that is all
# this fast tier has to go on -- the same limitation check_spec_strategy
# already accepts for parameter-count inference.
_VISION_MODEL_TOKENS = ("vl", "vision", "llava", "moondream", "minicpm-v", "pixtral", "bakllava")


def check_equipment_conformance_capability(settings: dict[str, Any]) -> Check:
    """Is the configured reviewer actually able to run the D1
    equipment-duplication guard and post-render conformance check
    (spec_conformance.py and duplicate_detection.py)?

    Both call `StudioQwen.visual_presence`, which always exists as a method
    regardless of the underlying model -- so a text-only model configured as
    the reviewer does not raise an error, it just answers an image question
    it cannot see, silently. That is a worse failure than a missing method:
    the exact "one shield became two and nobody caught it" defect class this
    mechanism exists to catch would go uncaught with no error anywhere.
    """
    model = str(settings.get("model", "")).lower()
    if not model:
        return Check(name="equipment-conformance vision capability", status="skip", detail="no model configured")
    if any(token in model for token in _VISION_MODEL_TOKENS):
        return Check(
            name="equipment-conformance vision capability",
            status="ok",
            detail=f"{model} looks vision-capable by name",
        )
    return Check(
        name="equipment-conformance vision capability",
        status="warn",
        detail=(
            f"{model} does not look vision-capable by name. D1's shield-repair guard and "
            "check_equipment_conformance() both ask this model image-grounded yes/no questions; "
            "a text-only model answers anyway, with no error, and duplicate equipment will not "
            "be caught."
        ),
        remedy="Set [studio_defaults].model to a vision-capable id (contains 'vl', 'vision', "
        "'llava', etc.) -- the same model already used for D1-D9 image review qualifies.",
    )


def check_llm_endpoint(settings: dict[str, Any]) -> Check:
    url = str(settings.get("localdeploy_url", "")).rstrip("/")
    model = str(settings.get("model", ""))
    if not url:
        return Check(name="LLM endpoint", status="skip", detail="no localdeploy_url configured")
    payload = _http_json(url + "/models")
    if payload is None:
        return Check(
            name="LLM endpoint",
            status="fail",
            detail=f"{url}/models is not reachable",
            remedy="Start Ollama or LocalDeploy; D0 and every review gate need it.",
        )
    available = {str(item.get("id")) for item in (payload.get("data") or [])}
    if model and model not in available:
        return Check(
            name="LLM endpoint",
            status="fail",
            detail=f"{model!r} is not served by {url}",
            remedy=f"ollama pull {model}   (served: {', '.join(sorted(available)[:6]) or 'none'})",
        )
    return Check(name="LLM endpoint", status="ok", detail=f"{url} serving {model}")



# A vision review sends two images plus the spec and history. Measured on a
# real D3 review: 4,197 tokens. Ollama's default served context is 4,096, so
# the request is refused with an HTTP 400 that names a token count and
# nothing else -- several stages into a run, after the GPU work is done.
MINIMUM_REVIEWER_CONTEXT = 8192

# ...and an upper bound, because context is not free. The KV cache is part of
# the model's resident footprint, so raising the window raises VRAM: measured
# on the 4B reviewer, 4,096 tokens cost 3.6 GB and 16,384 cost 5.09 GB. It
# still "fit" at 16,384 -- ollama reported it fully on GPU -- but it left only
# ~1.1 GB for the vision encoder's working memory on an 8 GB card, and the
# same D4 review that takes 49 s at 8,192 ran past 600 s. Fitting and being
# fast are different properties; this bound protects the second one.
MAXIMUM_REVIEWER_CONTEXT_SMALL_CARD = 8192


def check_reviewer_context(settings: dict[str, Any]) -> Check:
    """Is the reviewer *served* with enough context for a vision review?

    Distinct from what the model supports: qwen3-vl is trained to 262,144
    tokens but Ollama serves 4,096 unless told otherwise, and the shortfall
    only shows up as a 400 from a stage that already spent its GPU budget.
    """
    url = str(settings.get("localdeploy_url", "")).rstrip("/")
    model = str(settings.get("model", ""))
    if not url or not model:
        return Check(name="reviewer context window", status="skip", detail="no endpoint configured")
    root = url[: -len("/v1")] if url.endswith("/v1") else url
    served: int | None = None
    running = _http_json(root + "/api/ps", timeout=6)
    if isinstance(running, dict):
        for item in running.get("models") or []:
            if str(item.get("name")) == model and item.get("context_length"):
                served = int(item["context_length"])
    if served is None:
        return Check(
            name="reviewer context window",
            status="warn",
            detail=f"{model} is not loaded, so its served context could not be read",
            remedy=f"Run one request to load it, then re-check. It must serve at least "
            f"{MINIMUM_REVIEWER_CONTEXT} tokens.",
        )
    if served < MINIMUM_REVIEWER_CONTEXT:
        return Check(
            name="reviewer context window",
            status="fail",
            detail=(
                f"{model} is served with only {served} tokens of context. A vision review "
                "(two images plus the spec and history) measured 4,197 tokens and is refused "
                "with HTTP 400 mid-run."
            ),
            remedy=f"Set OLLAMA_CONTEXT_LENGTH={MINIMUM_REVIEWER_CONTEXT} and restart Ollama. "
            "The model itself supports far more; this is only the served window.",
        )
    hardware = detect_hardware(url)
    total = hardware.vram_total_gb or 0
    if total and total < 12 and served > MAXIMUM_REVIEWER_CONTEXT_SMALL_CARD:
        return Check(
            name="reviewer context window",
            status="warn",
            detail=(
                f"{model} is served with {served} tokens on a {total:.0f} GB card. The KV cache "
                "is resident VRAM, so a large window crowds out the vision encoder's working "
                "memory: measured, the same review took 49 s at 8,192 and over 600 s at 16,384."
            ),
            remedy=f"Set OLLAMA_CONTEXT_LENGTH={MAXIMUM_REVIEWER_CONTEXT_SMALL_CARD} and restart Ollama.",
        )
    return Check(
        name="reviewer context window",
        status="ok",
        detail=f"{model} served with {served} tokens",
    )


# What each concept backend needs resident, mirroring
# studio_pipeline._VramHandoff._COMFY_REQUIRED_GB. Duplicated deliberately:
# preflight must not import the coordinator, and a drift between the two is
# caught by test_concept_backend_envelopes_match_the_admission_gate.
CONCEPT_BACKEND_VRAM_GB = {
    "z_image_turbo": 6.5,
    "qwen_image_2512": 6.5,
    "qwen_image_edit_2511": 6.5,
    "sdxl": 5.8,
}


def check_concept_backend_fits(hardware: HardwareProfile, settings: dict[str, Any]) -> Check:
    """Can the selected image backend ever be admitted on this machine?

    The gap this closes was expensive to find the hard way. Every other
    memory check here asks about the *reviewer*, which is the smallest
    model in the pipeline; nothing asked whether the image backend fits.
    On a laptop GPU it usually does not: a Windows desktop with a browser,
    an editor, and a vendor overlay can hold 2-3 GB of an 8 GB card before
    any model loads, and every concept backend needs 5.8-6.5 GB plus the
    safety margin. The run then compiles its contract at D0, reaches D1,
    waits out the full unload timeout, and fails with "GPU memory did not
    recover" -- a message about a symptom, minutes after the point where
    the answer was already knowable.

    Reported against *live* free memory plus whatever the reviewer would
    release, because that is the real ceiling: total capacity is not
    reachable while a desktop is running on the same card.
    """
    backend = str(settings.get("concept_backend", "") or "")
    if backend in {"", "auto"}:
        # "auto" resolves at run time against what is installed; the
        # cheapest candidate is the honest bound to check.
        required = min(CONCEPT_BACKEND_VRAM_GB.values())
        label = "the cheapest available backend"
    elif backend in CONCEPT_BACKEND_VRAM_GB:
        required = CONCEPT_BACKEND_VRAM_GB[backend]
        label = backend
    else:
        return Check(
            name="concept backend fits in VRAM",
            status="skip",
            detail=f"no measured envelope for concept_backend {backend!r}",
        )

    margin = float(settings.get("gpu_safety_margin_gb", 0.75) or 0.75)
    needed = required + margin
    total = hardware.vram_total_gb
    if not total:
        return Check(
            name="concept backend fits in VRAM",
            status="skip",
            detail="no GPU telemetry, so the image backend's fit cannot be proven",
        )

    snapshot = gpu_memory_snapshot()
    device = (snapshot or {}).get("device") or {}
    free_now = float(device.get("free_gb") or 0)
    # The reviewer is unloaded before an image job, so whatever it holds
    # *right now* is reclaimable. Measured from the server rather than
    # assumed from the model name: adding a model's nominal size back while
    # it is not actually resident double-counts memory that was never taken
    # and reports a machine as ready when it is not.
    reclaimable = 0.0
    reviewer_root = str(settings.get("localdeploy_url", "")).rstrip("/")
    if reviewer_root.endswith("/v1"):
        reviewer_root = reviewer_root[: -len("/v1")]
    if reviewer_root:
        running = _http_json(reviewer_root + "/api/ps", timeout=4)
        if isinstance(running, dict):
            for item in running.get("models") or []:
                resident = item.get("size_vram")
                if isinstance(resident, (int, float)):
                    reclaimable += float(resident) / (1024**3)
    reachable = min(total, free_now + reclaimable) if free_now else total

    if needed > total:
        return Check(
            name="concept backend fits in VRAM",
            status="fail",
            detail=(
                f"{label} needs {needed:.2f} GB but the card only has {total:.2f} GB in total"
            ),
            remedy="Choose a smaller concept backend; this one cannot run on this GPU at any desktop load.",
        )
    if free_now and needed > reachable:
        held = max(0.0, total - free_now - reclaimable)
        return Check(
            name="concept backend fits in VRAM",
            status="fail",
            detail=(
                f"{label} needs {needed:.2f} GB free, but this machine can currently reach only "
                f"{reachable:.2f} GB ({total:.2f} GB total, {held:.2f} GB held by other processes). "
                "Every D1 attempt will wait out the unload timeout and then fail."
            ),
            remedy=(
                "Close GPU-using applications -- a browser, an editor, or a vendor overlay are the "
                f"usual holders of {held:.2f} GB -- or select a concept backend with a smaller envelope."
            ),
        )
    return Check(
        name="concept backend fits in VRAM",
        status="ok",
        detail=f"{label} needs {needed:.2f} GB; {reachable:.2f} GB is reachable on this machine",
    )


COMFY_CPU_OFFLOAD_FLAGS = ("--cpu", "--novram", "--lowvram")


def check_comfy_memory_mode(settings: dict[str, Any]) -> Check:
    """Is ComfyUI keeping weights on the GPU, or streaming them from RAM?

    This is the failure mode with no error message. A ComfyUI launched with
    --lowvram still renders every image correctly, so nothing reports a
    problem -- it just moves weights across PCIe for each pass. Measured on
    this pipeline: 8m52s of model initialization to produce 30s of
    sampling, against 45s end to end for the identical render with the
    model resident. A user watching that concludes the tool is broken, and
    no log line disagrees with them.

    Reported as a failure under a GPU-only device policy, because that
    policy exists precisely to forbid silent CPU inference, and as a
    warning otherwise, where the trade is a legitimate choice.
    """
    url = str(settings.get("comfy_url", "")).rstrip("/")
    if not url:
        return Check(name="ComfyUI keeps weights on the GPU", status="skip", detail="no comfy_url configured")
    payload = _http_json(url + "/system_stats", timeout=6)
    if payload is None:
        return Check(
            name="ComfyUI keeps weights on the GPU",
            status="skip",
            detail=f"{url} is not reachable, so its memory mode is unknown",
        )
    argv = [str(item) for item in (payload.get("system") or {}).get("argv") or []]
    offload = [flag for flag in COMFY_CPU_OFFLOAD_FLAGS if flag in argv]
    if not offload:
        return Check(
            name="ComfyUI keeps weights on the GPU",
            status="ok",
            detail="no CPU-offload flags; weights stay resident on the device",
        )
    strict = str(settings.get("device_policy", "")) in {"gpu_compute_only", "strict_device_only"}
    return Check(
        name="ComfyUI keeps weights on the GPU",
        status="fail" if strict else "warn",
        detail=(
            f"ComfyUI is running with {' '.join(offload)}, so it streams model weights from "
            "system RAM. Renders still succeed; they take minutes instead of seconds, with no "
            "error to explain why."
        ),
        remedy=(
            "Restart ComfyUI without those flags (--reserve-vram is fine and does not offload). "
            "If the card genuinely cannot hold the model, choose a smaller concept backend "
            "instead of streaming a large one."
        ),
    )


def check_comfy_nodes(settings: dict[str, Any], stages: dict[str, Any]) -> Check:
    url = str(settings.get("comfy_url", "")).rstrip("/")
    backend = str((stages.get("D2") or {}).get("backend", "hunyuan3d"))
    required = list(_D2_REQUIRED_NODES.get(backend) or ())
    if settings.get("vae_tiling") is True:
        required.append("VAEDecodeTiled")
    if not url:
        return Check(name="ComfyUI nodes for D2", status="skip", detail="no comfy_url configured")
    if not required:
        return Check(
            name="ComfyUI nodes for D2",
            status="skip",
            detail=f"D2 backend {backend!r} is a subprocess worker, not a ComfyUI graph",
        )
    # Per-node endpoints, not the whole /object_info: the full document is
    # several megabytes on a ComfyUI with custom nodes installed, which made
    # the System page take longer to render than the checks are worth.
    if _http_json(url + "/system_stats", timeout=6) is None:
        return Check(
            name="ComfyUI nodes for D2",
            status="fail",
            detail=f"{url} is not reachable",
            remedy=(
                "Start ComfyUI: python main.py --listen 127.0.0.1 --port 8188 "
                "--normalvram --reserve-vram 0.75 --preview-method none"
            ),
        )
    with ThreadPoolExecutor(max_workers=len(required)) as pool:
        found = dict(
            zip(
                required,
                pool.map(lambda node: _http_json(f"{url}/object_info/{node}", timeout=8), required),
            )
        )
    missing = [node for node, payload in found.items() if not payload or node not in payload]
    if missing:
        return Check(
            name="ComfyUI nodes for D2",
            status="fail",
            detail=f"backend {backend!r} needs {missing}, which this ComfyUI does not have",
            remedy="Update ComfyUI (git pull + pip install -r requirements.txt). Native "
            "Hunyuan3D support post-dates late-2024 builds.",
        )
    return Check(
        name="ComfyUI nodes for D2", status="ok", detail=f"all {len(required)} nodes present for {backend}"
    )


def check_comfy_checkpoints(settings: dict[str, Any], stages: dict[str, Any]) -> Check:
    url = str(settings.get("comfy_url", "")).rstrip("/")
    if not url:
        return Check(name="ComfyUI checkpoints", status="skip", detail="no comfy_url configured")
    payload = _http_json(url + "/object_info/CheckpointLoaderSimple", timeout=25)
    if payload is None:
        return Check(
            name="ComfyUI checkpoints",
            status="fail",
            detail=f"{url} is not reachable",
            remedy="Start ComfyUI.",
        )
    try:
        installed = set(
            payload["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        )
    except Exception:
        return Check(
            name="ComfyUI checkpoints",
            status="warn",
            detail="could not read the checkpoint list from object_info",
        )
    wanted = {
        "D1 concept": str(settings.get("checkpoint", "")),
        "D2 geometry": str((stages.get("D2") or {}).get("checkpoint", "")),
    }
    missing = {label: name for label, name in wanted.items() if name and name not in installed}
    if missing:
        return Check(
            name="ComfyUI checkpoints",
            status="fail",
            detail="; ".join(f"{label} wants {name!r}" for label, name in missing.items()),
            remedy="Download the file into ComfyUI/models/checkpoints/, or point the config at "
            f"one you have ({', '.join(sorted(installed)[:3])}...).",
        )
    return Check(
        name="ComfyUI checkpoints",
        status="ok",
        detail=", ".join(f"{label}={name}" for label, name in wanted.items() if name),
    )


_Z_IMAGE_TURBO_REQUIRED_FILES = {
    "UNETLoader": ("unet_name", "z_image_turbo_int8_convrot.safetensors"),
    "CLIPLoader": ("clip_name", "qwen_3_4b_fp8_mixed.safetensors"),
    "VAELoader": ("vae_name", "ae.safetensors"),
}


def check_z_image_turbo_models(settings: dict[str, Any]) -> Check:
    """Are Z-Image Turbo's three model files actually installed in ComfyUI?

    Only actionable -- and only run -- when concept_backend is explicitly
    z_image_turbo; a card not using this backend has no reason to fail
    preflight over files it will never load. Mirrors check_comfy_checkpoints'
    object_info-per-loader-node idiom rather than the pipeline's live
    comfy.models(...) call, so this works without a running StudioComfyClient.
    """
    url = str(settings.get("comfy_url", "")).rstrip("/")
    if str(settings.get("concept_backend", "")) != "z_image_turbo":
        return Check(
            name="Z-Image Turbo models",
            status="skip",
            detail="concept_backend is not z_image_turbo",
        )
    if not url:
        return Check(name="Z-Image Turbo models", status="skip", detail="no comfy_url configured")
    missing = []
    for node, (input_key, filename) in _Z_IMAGE_TURBO_REQUIRED_FILES.items():
        payload = _http_json(f"{url}/object_info/{node}", timeout=8)
        try:
            installed = set(payload[node]["input"]["required"][input_key][0])
        except Exception:
            return Check(
                name="Z-Image Turbo models",
                status="fail",
                detail=f"{url} is not reachable",
                remedy=(
                    "Start ComfyUI: python main.py --listen 127.0.0.1 --port 8188 "
                    "--normalvram --reserve-vram 0.75 --preview-method none"
                ),
            )
        if filename not in installed:
            missing.append(filename)
    if missing:
        return Check(
            name="Z-Image Turbo models",
            status="fail",
            detail=f"missing {missing}",
            remedy="Run resources/adapters/install_qwen_image_edit_models.py "
            "--profile z-image-turbo.",
        )
    return Check(name="Z-Image Turbo models", status="ok", detail="all 3 files present")


def check_voxel_vs_grip(stages: dict[str, Any], asset_height_m: float = 1.8) -> Check:
    d3 = stages.get("D3") or {}
    if str(d3.get("manifold_policy", "weld_only")) == "weld_only":
        return Check(
            name="voxel size keeps fingers separate",
            status="skip",
            detail="manifold_policy=weld_only, so no remesh runs",
        )
    fraction = float(d3.get("voxel_fraction", 0.006))
    voxel_mm = fraction * asset_height_m * 1000
    limit_mm = FINGER_WIDTH_M * 1000 / 2
    if voxel_mm > limit_mm:
        return Check(
            name="voxel size keeps fingers separate",
            status="fail",
            detail=(
                f"voxel_fraction={fraction} is {voxel_mm:.1f} mm on a {asset_height_m:.2f} m asset; "
                f"fingers are ~{FINGER_WIDTH_M * 1000:.0f} mm and fuse above ~{limit_mm:.1f} mm"
            ),
            remedy=f"Set [stage_defaults.D3].voxel_fraction to {limit_mm / 1000 / asset_height_m:.4f} "
            "or lower, otherwise D7 fails with 'grip topology did not resolve the expected "
            "digit branches: []' long after D3 reported success.",
        )
    return Check(
        name="voxel size keeps fingers separate",
        status="ok",
        detail=f"{voxel_mm:.1f} mm voxels vs ~{FINGER_WIDTH_M * 1000:.0f} mm fingers",
    )


def check_donor_motion(stages: dict[str, Any], repo_root: Path) -> Check:
    clip_id = str((stages.get("D7") or {}).get("donor_motion_id", "")).strip()
    if not clip_id:
        return Check(
            name="D7 donor motion",
            status="warn",
            detail="no donor_motion_id set; D7 falls back to a hardcoded path",
            remedy="Set [stage_defaults.D7].donor_motion_id to a clip in resources/motion_library/catalog.json.",
        )
    catalog = repo_root / "motion_library" / "catalog.json"
    if not catalog.is_file():
        return Check(
            name="D7 donor motion",
            status="fail",
            detail=f"donor_motion_id={clip_id!r} but resources/motion_library/catalog.json does not exist",
            remedy="Copy resources/motion_library/catalog.example.json and fill in a real CC0 clip.",
        )
    try:
        path = resolve_donor_motion_path(load_motion_library(catalog), clip_id, catalog_dir=catalog.parent)
    except Exception as exc:
        return Check(
            name="D7 donor motion",
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
            remedy="Fix the clip entry or download its file.",
        )
    if not path.is_file():
        return Check(
            name="D7 donor motion",
            status="fail",
            detail=f"catalog resolves {clip_id!r} to {path}, which does not exist",
            remedy="Download the clip to that path.",
        )
    return Check(name="D7 donor motion", status="ok", detail=f"{clip_id} -> {path.name}")


# ---- deep checks: these launch Blender, so they cost seconds, not milliseconds


_BLENDER_PROBE = r"""
import json, sys
import bpy
engines = list(bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys())
result = {"version": bpy.app.version_string, "engines": engines}
source = None
for index, value in enumerate(sys.argv):
    if value == "--donor" and index + 1 < len(sys.argv):
        source = sys.argv[index + 1]
if source:
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=source)
        armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        result["donor_bones"] = sorted(b.name for b in armatures[0].data.bones) if armatures else []
        result["donor_actions"] = sorted(a.name for a in bpy.data.actions)
    except Exception as exc:
        result["donor_error"] = f"{type(exc).__name__}: {exc}"
print("PREFLIGHT_JSON " + json.dumps(result))
"""


def _run_blender_probe(blender: Path, donor: Path | None) -> dict[str, Any] | None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "probe.py"
        script.write_text(_BLENDER_PROBE, encoding="utf-8")
        command = [
            str(blender),
            "--background",
            "--factory-startup",
            "--offline-mode",
            "--python",
            str(script),
        ]
        if donor:
            command += ["--", "--donor", str(donor)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except Exception:
            return None
    for line in result.stdout.splitlines():
        if line.startswith("PREFLIGHT_JSON "):
            try:
                return json.loads(line[len("PREFLIGHT_JSON ") :])
            except Exception:
                return None
    return None


# What resources/adapters/*.py actually assign to scene.render.engine, and the
# bone names resources/adapters/retarget_humanoid_motion.py contracts against.
_REQUIRED_ENGINES = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
_REQUIRED_DONOR_BONES = (
    "pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "Head",
    "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r",
    "lowerarm_l", "lowerarm_r", "hand_l", "hand_r",
    "thigh_l", "thigh_r", "calf_l", "calf_r",
    "foot_l", "foot_r", "ball_l", "ball_r",
)
_REQUIRED_DONOR_ACTIONS = ("Idle_Loop", "Walk_Loop", "Sword_Attack", "Hit_Chest", "Death01")


def deep_blender_checks(stages: dict[str, Any], repo_root: Path) -> list[Check]:
    config = load_local_config()
    binding = worker_binding(config, "blender") if config else None
    executable = None
    if binding and binding.command_prefix:
        executable = Path(binding.command_prefix[0])
    if executable is None or not executable.is_file():
        found = shutil.which("blender")
        executable = Path(found) if found else None
    if executable is None:
        return [
            Check(
                name="Blender render engine",
                status="fail",
                detail="no Blender binding in config.local.toml and none on PATH",
                remedy="Add [workers.blender].command_prefix to config.local.toml.",
            )
        ]

    donor: Path | None = None
    clip_id = str((stages.get("D7") or {}).get("donor_motion_id", "")).strip()
    catalog = repo_root / "motion_library" / "catalog.json"
    if clip_id and catalog.is_file():
        try:
            candidate = resolve_donor_motion_path(
                load_motion_library(catalog), clip_id, catalog_dir=catalog.parent
            )
            donor = candidate if candidate.is_file() else None
        except Exception:
            donor = None

    probe = _run_blender_probe(executable, donor)
    if probe is None:
        return [
            Check(
                name="Blender render engine",
                status="fail",
                detail=f"{executable} did not answer the probe",
                remedy="Check the Blender path in config.local.toml.",
            )
        ]

    checks: list[Check] = []
    engines = set(probe.get("engines") or [])
    usable = [name for name in _REQUIRED_ENGINES if name in engines]
    checks.append(
        Check(
            name="Blender render engine",
            status="ok" if usable else "fail",
            detail=f"Blender {probe.get('version')} offers {sorted(engines)}"
            if not usable
            else f"Blender {probe.get('version')} provides {usable[0]}",
            remedy="" if usable else "The diagnostic renderers assign an engine name this build "
            "does not have; adapters must select from the build's own enum.",
        )
    )

    if donor is None:
        checks.append(
            Check(
                name="donor motion matches the retarget contract",
                status="skip",
                detail="no resolvable donor file to inspect",
            )
        )
        return checks

    if probe.get("donor_error"):
        checks.append(
            Check(
                name="donor motion matches the retarget contract",
                status="fail",
                detail=str(probe["donor_error"]),
            )
        )
        return checks

    bones = set(probe.get("donor_bones") or [])
    actions = set(probe.get("donor_actions") or [])
    missing_bones = [name for name in _REQUIRED_DONOR_BONES if name not in bones]
    missing_actions = [name for name in _REQUIRED_DONOR_ACTIONS if name not in actions]
    if missing_bones or missing_actions:
        checks.append(
            Check(
                name="donor motion matches the retarget contract",
                status="fail",
                detail=(
                    f"missing bones {missing_bones[:6]}"
                    + (f" and actions {missing_actions}" if missing_actions else "")
                ),
                remedy="Rename the donor skeleton onto the Unreal names the retarget adapter "
                "contracts against: blender --background --python "
                "resources/adapters/prepare_donor_motion.py -- --source <in> --output <out>",
            )
        )
    else:
        checks.append(
            Check(
                name="donor motion matches the retarget contract",
                status="ok",
                detail=f"{len(bones)} bones, {len(actions)} actions, all required names present",
            )
        )
    return checks


def run_preflight(
    *,
    profile: str = "simple",
    deep: bool = False,
    repo_root: Path | None = None,
    asset_height_m: float = 1.8,
) -> tuple[HardwareProfile, list[Check]]:
    """Every cross-stage assumption, checked concurrently."""
    root = repo_root or resource_root()
    resolved = resolve_settings(profile=profile)
    settings = studio_overrides(resolved)
    stages = resolved.values.get("stages", {}) or {}
    hardware = detect_hardware(str(settings.get("localdeploy_url", "")) or None)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(check_reviewer_fits, hardware, settings),
            pool.submit(check_device_policy, hardware, settings, stages),
            pool.submit(check_gpu_safety_margin, hardware, settings),
            pool.submit(check_concept_backend_fits, hardware, settings),
            pool.submit(check_comfy_memory_mode, settings),
            pool.submit(check_spec_strategy, settings),
            pool.submit(check_llm_endpoint, settings),
            pool.submit(check_reviewer_context, settings),
            pool.submit(check_equipment_conformance_capability, settings),
            pool.submit(check_comfy_nodes, settings, stages),
            pool.submit(check_comfy_checkpoints, settings, stages),
            pool.submit(check_z_image_turbo_models, settings),
            pool.submit(check_voxel_vs_grip, stages, asset_height_m),
            pool.submit(check_donor_motion, stages, root),
        ]
        deep_future = pool.submit(deep_blender_checks, stages, root) if deep else None
        checks: list[Check] = []
        for future in futures:
            try:
                checks.append(future.result())
            except Exception as exc:
                checks.append(
                    Check(name="preflight check", status="fail", detail=f"{type(exc).__name__}: {exc}")
                )
        if deep_future is not None:
            try:
                checks.extend(deep_future.result())
            except Exception as exc:
                checks.append(
                    Check(name="Blender deep check", status="fail", detail=f"{type(exc).__name__}: {exc}")
                )
    return hardware, checks
