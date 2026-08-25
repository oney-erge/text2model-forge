"""Minimal qualified ComfyUI concept adapter and local control surface for Text2Model Studio."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageFilter


class StudioComfyError(RuntimeError):
    pass


def _vae_decode_node(samples: list[str | int], vae: list[str | int], *, tiled: bool) -> dict[str, Any]:
    if not tiled:
        return {"class_type": "VAEDecode", "inputs": {"samples": samples, "vae": vae}}
    return {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": samples,
            "vae": vae,
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 8,
        },
    }


def explicit_cpu_nodes(workflow: dict[str, Any]) -> list[str]:
    """Node ids that explicitly request CPU execution in an inference graph."""

    return [
        str(node_id)
        for node_id, node in workflow.items()
        if str((node.get("inputs") or {}).get("device", "")).lower() == "cpu"
    ]


def concept_workflow(
    *,
    checkpoint: str,
    positive: str,
    negative: str,
    seed: int,
    prefix: str,
    loras: list[tuple[str, float]] | None = None,
    control_guides: list[tuple[str, str, float, float, float]] | None = None,
    width: int = 768,
    height: int = 1024,
    steps: int = 30,
    cfg: float = 6.0,
    sampler_name: str = "dpmpp_2m",
    scheduler: str = "karras",
    tiled_vae: bool = False,
) -> dict[str, Any]:
    """Single-figure concept workflow: checkpoint -> optional LoRA chain -> optional pose/depth
    ControlNet chain -> sampler. One coherent generation, never a region-composited one."""
    workflow: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
    }
    model_ref: list[str | int] = ["1", 0]
    clip_ref: list[str | int] = ["1", 1]
    vae_ref: list[str | int] = ["1", 2]
    node_id = 2
    for lora_name, strength in loras or []:
        lora_id = str(node_id)
        workflow[lora_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip": strength,
            },
        }
        model_ref, clip_ref = [lora_id, 0], [lora_id, 1]
        node_id += 1
    positive_id, negative_id = str(node_id), str(node_id + 1)
    workflow[positive_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": positive}}
    workflow[negative_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": negative}}
    positive_ref: list[str | int] = [positive_id, 0]
    negative_ref: list[str | int] = [negative_id, 0]
    node_id += 2
    for model_name, image_name, strength, start_percent, end_percent in control_guides or []:
        image_id, loader_id, apply_id = str(node_id), str(node_id + 1), str(node_id + 2)
        workflow[image_id] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        workflow[loader_id] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": model_name},
        }
        workflow[apply_id] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": positive_ref,
                "negative": negative_ref,
                "control_net": [loader_id, 0],
                "image": [image_id, 0],
                "vae": vae_ref,
                "strength": strength,
                "start_percent": start_percent,
                "end_percent": end_percent,
            },
        }
        positive_ref, negative_ref = [apply_id, 0], [apply_id, 1]
        node_id += 3
    latent_id, sampler_id, decode_id, save_id = (str(node_id + offset) for offset in range(4))
    workflow[latent_id] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    workflow[sampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": positive_ref,
            "negative": negative_ref,
            "latent_image": [latent_id, 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }
    workflow[decode_id] = _vae_decode_node([sampler_id, 0], vae_ref, tiled=tiled_vae)
    workflow[save_id] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": prefix, "images": [decode_id, 0]},
    }
    return workflow


def qwen_image_edit_2511_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    prefix: str,
    source_image: str,
    diffusion_model: str = "qwen_image_edit_2511_fp8mixed.safetensors",
    text_encoder: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    vae: str = "qwen_image_vae.safetensors",
    tiled_vae: bool = False,
) -> dict[str, Any]:
    """Native Qwen-Image-Edit-2511 ComfyUI graph for one identity-preserving edit.

    The source image is encoded through both Qwen-VL semantic conditioning and
    the VAE appearance path, matching the official Qwen/ComfyUI workflow.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 3.1},
        },
        "3": {"class_type": "CFGNorm", "inputs": {"model": ["2", 0], "strength": 1.0}},
        "4": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder, "type": "qwen_image", "device": "default"},
        },
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "6": {"class_type": "LoadImage", "inputs": {"image": source_image}},
        "7": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["6", 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0,
                "resolution_steps": 1,
            },
        },
        "8": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {"clip": ["4", 0], "prompt": prompt, "vae": ["5", 0], "image": ["7", 0]},
        },
        "9": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "clip": ["4", 0],
                "prompt": negative_prompt or " ",
                "vae": ["5", 0],
                "image": ["7", 0],
            },
        },
        "10": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["8", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "11": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["9", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["5", 0]}},
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["10", 0],
                "negative": ["11", 0],
                "latent_image": ["12", 0],
                "seed": seed,
                "steps": 40,
                "cfg": 4.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "14": _vae_decode_node(["13", 0], ["5", 0], tiled=tiled_vae),
        "15": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["14", 0]}},
    }


def z_image_turbo_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    prefix: str,
    width: int = 1104,
    height: int = 1472,
    steps: int = 8,
    cfg: float = 1.0,
    diffusion_model: str = "z_image_turbo_int8_convrot.safetensors",
    text_encoder: str = "qwen_3_4b_fp8_mixed.safetensors",
    vae: str = "ae.safetensors",
    tiled_vae: bool = False,
) -> dict[str, Any]:
    """Native Z-Image Turbo text-to-image graph for D1 concept creation.

    Z-Image Turbo is a distilled 6B model: 8 steps and cfg=1 (no real negative
    guidance) is the published turbo setting, not an arbitrary default -- more
    steps or higher cfg does not track the model's own training regime the
    way it does for a non-distilled model. `text_encoder` is state-dict-
    detected by ComfyUI's CLIPLoader as Qwen3-4B and auto-routed to the
    Z-Image text encoder regardless of a declared `type=`, so no custom node
    or ComfyUI-GGUF install is required -- confirmed against the installed
    ComfyUI 0.33.0's comfy/sd.py detect_te_model path.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder, "type": "stable_diffusion", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt or " ", "clip": ["2", 0]},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": _vae_decode_node(["7", 0], ["3", 0], tiled=tiled_vae),
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }


def qwen_image_2512_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    prefix: str,
    width: int = 1104,
    height: int = 1472,
    diffusion_model: str = "qwen_image_2512_fp8_e4m3fn.safetensors",
    text_encoder: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    vae: str = "qwen_image_vae.safetensors",
    tiled_vae: bool = False,
) -> dict[str, Any]:
    """Official-style native Qwen-Image-2512 text-to-image graph for D1 concept creation.

    D1 deliberately starts with the text-to-image foundation model. Qwen Image Edit
    is reserved for a later bounded repair where a real, already-approved image exists.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.1}},
        "3": {"class_type": "CFGNorm", "inputs": {"model": ["2", 0], "strength": 1.0}},
        "4": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder, "type": "qwen_image", "device": "default"},
        },
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 0]}},
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt or " ", "clip": ["4", 0]},
        },
        "8": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": 50,
                "cfg": 4.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "10": _vae_decode_node(["9", 0], ["5", 0], tiled=tiled_vae),
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["10", 0]}},
    }


def inpaint_workflow(
    *,
    checkpoint: str,
    positive: str,
    negative: str,
    seed: int,
    prefix: str,
    source_image: str,
    mask_image: str,
    denoise: float,
    loras: list[tuple[str, float]] | None = None,
    tiled_vae: bool = False,
) -> dict[str, Any]:
    """Core-node local repair for an ordinary (non-inpaint) checkpoint.

    Applies the same LoRA chain as the full-body pass so a local repair (e.g. adding a missing
    shield) stays in the same rendered style as the rest of the figure.
    """
    workflow: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
    }
    model_ref: list[str | int] = ["1", 0]
    clip_ref: list[str | int] = ["1", 1]
    vae_ref: list[str | int] = ["1", 2]
    node_id = 2
    for lora_name, strength in loras or []:
        lora_id = str(node_id)
        workflow[lora_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip": strength,
            },
        }
        model_ref, clip_ref = [lora_id, 0], [lora_id, 1]
        node_id += 1
    positive_id, negative_id, image_id, mask_id, encode_id, diff_id, sampler_id, decode_id, save_id = (
        str(node_id + offset) for offset in range(9)
    )
    workflow[positive_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": positive}}
    workflow[negative_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": negative}}
    workflow[image_id] = {"class_type": "LoadImage", "inputs": {"image": source_image}}
    workflow[mask_id] = {"class_type": "LoadImageMask", "inputs": {"image": mask_image, "channel": "red"}}
    workflow[encode_id] = {
        "class_type": "VAEEncodeForInpaint",
        "inputs": {
            "pixels": [image_id, 0],
            "vae": vae_ref,
            "mask": [mask_id, 0],
            "grow_mask_by": 12,
        },
    }
    workflow[diff_id] = {"class_type": "DifferentialDiffusion", "inputs": {"model": model_ref}}
    workflow[sampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "model": [diff_id, 0],
            "positive": [positive_id, 0],
            "negative": [negative_id, 0],
            "latent_image": [encode_id, 0],
            "seed": seed,
            "steps": 14,
            "cfg": 3.5,
            "sampler_name": "dpmpp_sde",
            "scheduler": "karras",
            "denoise": denoise,
        },
    }
    workflow[decode_id] = _vae_decode_node([sampler_id, 0], vae_ref, tiled=tiled_vae)
    workflow[save_id] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": prefix, "images": [decode_id, 0]},
    }
    return workflow


def make_humanoid_openpose_guide(output_directory: Path) -> Path:
    """Create a deterministic OpenPose-like guide for a frontal humanoid."""
    output_directory.mkdir(parents=True, exist_ok=True)
    width, height = 768, 1024
    pose = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(pose)
    points = {
        "nose": (384, 150),
        "neck": (384, 242),
        "right_shoulder": (304, 276),
        "right_elbow": (252, 400),
        "right_wrist": (216, 520),
        "left_shoulder": (464, 276),
        "left_elbow": (520, 402),
        "left_wrist": (536, 510),
        "right_hip": (342, 542),
        "right_knee": (330, 710),
        "right_ankle": (316, 888),
        "left_hip": (426, 542),
        "left_knee": (438, 710),
        "left_ankle": (452, 888),
    }
    limbs = [
        ("neck", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("neck", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("neck", "right_hip"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("neck", "left_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("neck", "nose"),
    ]
    colors = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (128, 255, 0),
        (0, 255, 0),
        (0, 255, 128),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
        (128, 0, 255),
        (255, 0, 255),
        (255, 0, 128),
        (255, 80, 80),
    ]
    for index, (start, end) in enumerate(limbs):
        draw.line((points[start], points[end]), fill=colors[index], width=11)
    for point in points.values():
        draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill="white")
    pose_path = output_directory / "openpose_layout.png"
    pose.save(pose_path)
    return pose_path


def make_humanoid_equipment_layout_guide(output: Path) -> Path:
    """Draw a softly rendered layout reference for Qwen Image Edit.

    It encodes only one body and the equipment sides.  It deliberately avoids hard
    outlines, pixels, and block forms so the model is not invited to preserve a
    retro-sprite style from its source image.
    """
    width, height, scale = 768, 1024, 3
    size = (width * scale, height * scale)
    canvas = Image.new("RGB", size, "#e8e4dc")
    background = ImageDraw.Draw(canvas)
    for y in range(size[1]):
        shade = 232 - int(13 * y / size[1])
        background.line((0, y, size[0], y), fill=(shade, shade - 3, shade - 8))
    art = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(art)
    point = lambda x, y: (x * scale, y * scale)
    box = lambda x0, y0, x1, y1: (x0 * scale, y0 * scale, x1 * scale, y1 * scale)

    # Soft floor shadow only: never a wheel base, platform, or pedestal.
    shadow = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(box(254, 900, 528, 956), fill=(55, 43, 35, 48))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(22 * scale)))

    steel_dark, steel, steel_light = "#46525d", "#6f7e88", "#b5c0c5"
    cloth, leather, skin = "#315b8c", "#744c34", "#c99270"
    # A single connected head, neck, torso, two arms, and two legs.
    draw.rounded_rectangle(box(362, 190, 406, 268), radius=16 * scale, fill=steel_dark)
    draw.ellipse(box(324, 110, 444, 252), fill="#8c9aa1")
    draw.ellipse(box(344, 138, 424, 232), fill=skin)
    draw.pieslice(box(324, 108, 444, 246), 190, 355, fill=steel_dark)
    draw.polygon([point(300, 238), point(468, 238), point(450, 566), point(318, 566)], fill=cloth)
    draw.rounded_rectangle(box(296, 246, 472, 388), radius=32 * scale, fill=steel)
    draw.rounded_rectangle(box(312, 262, 456, 368), radius=24 * scale, fill=steel_light)
    # Right arm / viewer-left, with one compact closed hand at the hilt.
    draw.line((point(314, 290), point(258, 364), point(222, 470)), fill=steel_dark, width=70 * scale, joint="curve")
    draw.line((point(314, 290), point(258, 364), point(222, 470)), fill=steel, width=52 * scale, joint="curve")
    draw.ellipse(box(194, 442, 246, 496), fill=skin)
    # Left arm / viewer-right, entering the shield straps.
    draw.line((point(454, 290), point(506, 366), point(544, 464)), fill=steel_dark, width=70 * scale, joint="curve")
    draw.line((point(454, 290), point(506, 366), point(544, 464)), fill=steel, width=52 * scale, joint="curve")
    draw.ellipse(box(520, 438, 570, 494), fill=skin)
    # A normal straight sword: blade, guard, wrapped hilt, and pommel.
    draw.polygon([point(208, 454), point(151, 154), point(174, 148), point(234, 455)], fill="#c6d0d2")
    draw.polygon([point(215, 448), point(161, 168), point(173, 164), point(226, 450)], fill="#f3f5f2")
    draw.rounded_rectangle(box(181, 434, 252, 450), radius=5 * scale, fill=steel_dark)
    draw.line((point(221, 452), point(236, 516)), fill=leather, width=20 * scale)
    draw.ellipse(box(222, 506, 250, 532), fill=steel_dark)
    # One medium, opaque shield, deliberately distinct from the body.
    draw.rounded_rectangle(box(492, 338, 686, 682), radius=48 * scale, fill=steel_dark)
    draw.rounded_rectangle(box(504, 350, 674, 670), radius=40 * scale, fill="#c39a68")
    draw.rounded_rectangle(box(518, 366, 660, 654), radius=32 * scale, fill="#9c714c")
    draw.line((point(530, 454), point(650, 478)), fill=leather, width=14 * scale)
    draw.line((point(526, 522), point(646, 546)), fill=leather, width=14 * scale)
    # Two separate, natural legs and grounded boots.
    draw.line((point(350, 548), point(328, 734), point(316, 878)), fill=steel_dark, width=76 * scale, joint="curve")
    draw.line((point(418, 548), point(440, 734), point(452, 878)), fill=steel_dark, width=76 * scale, joint="curve")
    draw.line((point(350, 548), point(328, 734), point(316, 878)), fill=steel, width=56 * scale, joint="curve")
    draw.line((point(418, 548), point(440, 734), point(452, 878)), fill=steel, width=56 * scale, joint="curve")
    draw.rounded_rectangle(box(274, 862, 350, 926), radius=24 * scale, fill="#3c454a")
    draw.rounded_rectangle(box(420, 862, 496, 926), radius=24 * scale, fill="#3c454a")
    draw.rounded_rectangle(box(286, 870, 346, 910), radius=18 * scale, fill=steel_light)
    draw.rounded_rectangle(box(424, 870, 484, 910), radius=18 * scale, fill=steel_light)

    canvas = Image.alpha_composite(canvas, art)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(output)
    return output


# Compatibility alias for integrations created before the public rename.
make_footman_equipment_layout_guide = make_humanoid_equipment_layout_guide


class StudioComfyClient:
    def __init__(self, base_url: str, *, timeout: float = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _failure(self, what: str, exc: Exception) -> StudioComfyError:
        """One message for every way a ComfyUI call can fail.

        "ComfyUI request failed for /models/controlnet: <urlopen error
        [WinError 10061] ...>" is what a real D1 produced on a machine
        without ComfyUI installed: it reads like an internal fault when the
        answer is simply that the service was never started. A refused
        connection is that case and nothing else, so it gets the remedy
        instead of the traceback.
        """
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ConnectionError):
            return StudioComfyError(
                f"ComfyUI is not running at {self.base_url}, so {what} could not complete. "
                "Start it with: python main.py --listen 127.0.0.1 --port 8188 "
                "--normalvram --reserve-vram 0.75 --preview-method none"
            )
        return StudioComfyError(f"ComfyUI request failed for {what}: {exc}")

    def _json(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                # Control endpoints are allowed to acknowledge with an empty
                # body.  Treat a successful empty response as success instead
                # of misreporting it as a JSON failure.
                return json.loads(body) if body.strip() else None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise self._failure(path, exc) from exc

    def health(self) -> dict[str, Any]:
        value = self._json("/system_stats")
        if not isinstance(value, dict):
            raise StudioComfyError("ComfyUI health response was not an object")
        return value

    def checkpoints(self) -> list[str]:
        return self.models("checkpoints")

    def models(self, kind: str) -> list[str]:
        value = self._json(f"/models/{urllib.parse.quote(kind)}")
        if not isinstance(value, list):
            raise StudioComfyError(f"ComfyUI {kind} response was not a list")
        return [str(item) for item in value]

    def controlnets(self) -> list[str]:
        return self.models("controlnet")

    def interrupt(self) -> None:
        """Ask local ComfyUI to stop its current workflow execution."""
        self._json("/interrupt", {})

    def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        """Release ComfyUI's model residency and allocator cache after work is idle."""
        self._json(
            "/free",
            {"unload_models": unload_models, "free_memory": free_memory},
        )

    def upload_image(self, name: str, data: bytes, subfolder: str = "text2model_studio") -> str:
        boundary = "----Text2ModelStudio" + uuid.uuid4().hex
        parts: list[bytes] = []
        for field, value in (("overwrite", "true"), ("type", "input"), ("subfolder", subfolder)):
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode(
                    "utf-8"
                )
            )
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode("utf-8")
        )
        parts.extend((data, f"\r\n--{boundary}--\r\n".encode("utf-8")))
        request = urllib.request.Request(
            self.base_url + "/upload/image",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise self._failure(f"uploading {name}", exc) from exc
        stored = str(value.get("name", name))
        stored_subfolder = str(value.get("subfolder", subfolder))
        return f"{stored_subfolder}/{stored}" if stored_subfolder else stored

    def generate(
        self,
        *,
        workflow: dict[str, Any],
        destination: Path,
        timeout_seconds: float = 900,
    ) -> list[Path]:
        submitted = self._json("/prompt", {"prompt": workflow})
        if not isinstance(submitted, dict) or submitted.get("node_errors"):
            raise StudioComfyError(f"ComfyUI rejected workflow: {submitted}")
        prompt_id = submitted.get("prompt_id")
        if not prompt_id:
            raise StudioComfyError("ComfyUI returned no prompt id")
        deadline = time.monotonic() + timeout_seconds
        history = None
        while time.monotonic() < deadline:
            value = self._json(f"/history/{urllib.parse.quote(str(prompt_id))}")
            if isinstance(value, dict) and prompt_id in value:
                candidate = value[prompt_id]
                status = candidate.get("status", {})
                if status.get("status_str") == "error":
                    raise StudioComfyError(f"ComfyUI job failed: {status}")
                if status.get("completed") is True:
                    history = candidate
                    break
            time.sleep(0.75)
        if history is None:
            raise StudioComfyError(f"ComfyUI timed out after {timeout_seconds}s")
        destination.mkdir(parents=True, exist_ok=True)
        outputs = []
        for node in history.get("outputs", {}).values():
            # Collect every downloadable artifact, not just "images". A 3D node
            # (Hunyuan3D's SaveGLB) reports its mesh under a different key --
            # "meshes"/"result"/"3d" depending on the node -- so key off the
            # shape of the record instead of a hardcoded list. Filtering on
            # "images" alone silently dropped every mesh ComfyUI produced.
            for records in node.values():
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict) or "filename" not in record:
                        continue
                    query = urllib.parse.urlencode(
                        {
                            "filename": record["filename"],
                            "subfolder": record.get("subfolder", ""),
                            "type": record.get("type", "output"),
                        }
                    )
                    target = destination / Path(record["filename"]).name
                    try:
                        with urllib.request.urlopen(
                            self.base_url + "/view?" + query, timeout=self.timeout
                        ) as response:
                            target.write_bytes(response.read())
                    except (urllib.error.URLError, TimeoutError) as exc:
                        raise StudioComfyError(
                            f"could not download {record['filename']}: {exc}"
                        ) from exc
                    outputs.append(target)
        if not outputs:
            raise StudioComfyError("ComfyUI completed without any downloadable output")
        return outputs


def hunyuan3d_workflow(
    *,
    image: str,
    prefix: str,
    seed: int,
    checkpoint: str = "hunyuan3d-dit-v2_fp16.safetensors",
    steps: int = 20,
    cfg: float = 8.0,
    octree_resolution: int = 256,
    latent_resolution: int = 3072,
) -> dict[str, Any]:
    """Single-image to 3D mesh using ComfyUI's NATIVE Hunyuan3D-2 support.

    This is the free, 8GB-class replacement for TRELLIS.2-4B, which needs
    16-24GB and therefore cannot run on a consumer laptop GPU at all. The
    mini checkpoint needs roughly 5GB for shape generation; the standard one
    about 6GB. No custom nodes are required -- ComfyUI ships these classes.

    `image` is the name returned by ComfyClient.upload_image(), so the mesh
    is generated from an image the human already approved rather than from a
    fresh render of a text prompt. That is the whole point: the approved 2D
    concept IS the 3D input.

    Texture generation is deliberately not requested here. ComfyUI's native
    support covers shape only, and the full shape+texture path needs ~12GB.
    Surface work stays in D8 where the pipeline already handles it.
    """
    return {
        "1": {
            "class_type": "ImageOnlyCheckpointLoader",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {"class_type": "LoadImage", "inputs": {"image": image}},
        # The checkpoint's slot 1 is the CLIP_VISION *model*; Hunyuan3D's
        # conditioning wants a CLIP_VISION_OUTPUT, so the image has to be
        # encoded through it first. crop="none" matches ComfyUI's own
        # Hunyuan3D image-to-model template.
        "10": {
            "class_type": "CLIPVisionEncode",
            "inputs": {"clip_vision": ["1", 1], "image": ["2", 0], "crop": "none"},
        },
        "3": {
            "class_type": "Hunyuan3Dv2Conditioning",
            "inputs": {"clip_vision_output": ["10", 0]},
        },
        # The DiT latent resolution (3072) is NOT the VAE's octree resolution
        # (256). Passing the octree value here produced a mesh of 53,560
        # disconnected fragments instead of a body.
        "4": {
            "class_type": "EmptyLatentHunyuan3Dv2",
            "inputs": {"resolution": latent_resolution, "batch_size": 1},
        },
        "5": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 1.0},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "positive": ["3", 0],
                "negative": ["3", 1],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "7": {
            "class_type": "VAEDecodeHunyuan3D",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2],
                "num_chunks": 8000,
                "octree_resolution": octree_resolution,
            },
        },
        # "surface net" produces a connected surface; the "basic" algorithm
        # emits per-voxel triangles that arrive as tens of thousands of
        # disconnected components and fail every downstream topology gate.
        "8": {
            "class_type": "VoxelToMesh",
            "inputs": {"voxel": ["7", 0], "algorithm": "surface net", "threshold": 0.6},
        },
        "9": {
            "class_type": "SaveGLB",
            "inputs": {"mesh": ["8", 0], "filename_prefix": prefix},
        },
    }


def make_chroma_alpha(
    source: Path,
    output: Path,
    *,
    tolerance: int = 46,
) -> dict[str, float | int | bool]:
    """Isolate the subject by growing the background inward from the border.

    Never uses a learned background-removal model: those ship gated,
    territory-restricted licences that this pipeline's lineage system
    correctly refuses for a release export.

    A fixed "is this pixel green?" test is not enough in practice. Diffusion
    models do not paint a flat chroma screen even when the prompt asks for
    one -- a real SDXL knight render came back with a vignetted background
    whose corners were (44,50,24) and (30,30,16), too dark and too grey to
    pass any green test, so the flood fill could not even seed there and half
    the background survived into the mesh.

    The safe fallback is deliberately narrower than learned segmentation:
    flood-fill only edge-connected pixels that retain measured green-screen
    channel ratios. A grey or blue result therefore fails closed instead of
    guessing where a similarly coloured subject ends. Reject or retry such a
    concept at D1; never promote an opaque or destructively guessed D2 seed.
    """
    with Image.open(source).convert("RGB") as image:
        width, height = image.size
        pixels = image.load()
        background = bytearray(width * height)

        def is_backdrop(x: int, y: int) -> bool:
            red, green, blue = pixels[x, y]
            # Green-over-BLUE is the reliable discriminator, and green-over-red
            # only has to be near-neutral. Measured on a real SDXL knight
            # render: backdrop pixels ran g/b 1.24-2.08 (including vignetted
            # corners as dark as (30,30,16)), while plate armour ran g/b
            # 0.98-1.32 and the red surcoat far below on g/r. Requiring
            # g >= r * 1.18, as this once did, rejected those dark corners --
            # the fill could not seed there and half the backdrop survived
            # into the mesh.
            return green >= blue * 1.20 and green >= red * 0.95

        queue: list[tuple[int, int]] = []
        for x in range(width):
            if is_backdrop(x, 0):
                queue.append((x, 0))
            if is_backdrop(x, height - 1):
                queue.append((x, height - 1))
        for y in range(height):
            if is_backdrop(0, y):
                queue.append((0, y))
            if is_backdrop(width - 1, y):
                queue.append((width - 1, y))

        # Edge connectivity is the safety mechanism, not a detail: the same
        # render has strongly green bounce-light on the armour -- (20,79,2)
        # and (50,68,21) -- which any purely per-pixel colour test would cut
        # holes through. Those pixels are enclosed by the subject, so a fill
        # seeded only from the border can never reach them.
        while queue:
            x, y = queue.pop()
            offset = y * width + x
            if background[offset] or not is_backdrop(x, y):
                continue
            background[offset] = 1
            if x:
                queue.append((x - 1, y))
            if x + 1 < width:
                queue.append((x + 1, y))
            if y:
                queue.append((x, y - 1))
            if y + 1 < height:
                queue.append((x, y + 1))

        alpha = Image.new("L", (width, height))
        alpha.putdata([0 if value else 255 for value in background])
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.8))
        result = image.convert("RGBA")
        # Flatten the removed backdrop to white in the COLOUR channels, not
        # just in alpha. Image-to-3D models read RGB through a CLIP-Vision
        # encoder that ignores the alpha channel entirely: leaving the old
        # green pixels behind at alpha=0 made Hunyuan3D reconstruct the
        # backdrop as a giant flat slab wrapped around the subject. Alpha is
        # still written for consumers that do respect it.
        white = Image.new("RGB", (width, height), (255, 255, 255))
        result = Image.composite(image, white, alpha)
        result = result.convert("RGBA")
        result.putalpha(alpha)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)
        alpha_values = alpha.get_flattened_data()
        transparent = sum(1 for value in alpha_values if value <= 8)
        opaque = sum(1 for value in alpha_values if value >= 247)
        total = width * height
        metrics = {
            "width": width,
            "height": height,
            "transparent_fraction": transparent / total,
            "opaque_fraction": opaque / total,
            "meaningful_alpha": transparent >= total * 0.01 and opaque >= total * 0.01,
        }
        if not metrics["meaningful_alpha"]:
            output.unlink(missing_ok=True)
            raise StudioComfyError(
                "geometry seed did not produce a safe edge-connected green screen; refusing opaque TRELLIS input"
            )
        return metrics
