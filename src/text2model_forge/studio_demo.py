"""Deterministic Studio-format tutorial data for the zero-model first run."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw
import trimesh

from .studio_models import (
    StudioAssetSpec,
    StudioComponent,
    StudioHumanDecision,
    StudioQwenReview,
)
from .studio_store import StudioStore


TUTORIAL_DESCRIPTION = (
    "A compact hand-painted cargo crate with reinforced corners, two recessed side handles, "
    "a sealed rectangular body, and a clean mobile-game silhouette."
)


def _preview(path: Path, *, accent: str, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (960, 720), "#11181c")
    draw = ImageDraw.Draw(image)
    draw.ellipse((190, 555, 780, 655), fill="#080d10")
    draw.polygon([(250, 250), (625, 205), (790, 310), (410, 365)], fill="#80664d")
    draw.polygon([(250, 250), (410, 365), (410, 585), (250, 450)], fill="#4f3b2d")
    draw.polygon([(410, 365), (790, 310), (790, 515), (410, 585)], fill="#684d38")
    for x in (280, 382, 690, 760):
        draw.rectangle((x, 285, x + 24, 545), fill=accent)
    draw.rectangle((515, 405, 665, 485), outline="#161b1e", width=18)
    draw.rectangle((532, 420, 648, 468), fill="#252d31")
    draw.line((250, 450, 410, 585, 790, 515), fill=accent, width=16)
    draw.text((42, 40), label, fill="#f4efe7")
    draw.text((42, 665), "Deterministic offline tutorial artifact", fill="#97a7ad")
    image.save(path)
    return path


def _mesh(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = trimesh.creation.box(extents=(1.2, 0.8, 0.72))
    body.apply_translation((0, 0, 0.36))
    body.export(path)
    return path


def create_tutorial_run(store: StudioStore, run_id: str):
    """Create a completed, clearly synthetic project visible in Studio.

    This is not a fake production success. Every evidence item carries a
    synthetic marker and the run itself has run_mode=tutorial so UI and API
    consumers can keep it separate from live and qualification evidence.
    """
    run = store.create(
        run_id,
        TUTORIAL_DESCRIPTION,
        {"profile": "simple", "run_mode": "tutorial"},
    )
    run.title = "Offline cargo-crate walkthrough"
    run.spec = StudioAssetSpec(
        asset_id="tutorial_cargo_crate",
        title=run.title,
        description=TUTORIAL_DESCRIPTION,
        creative_direction="Readable hand-painted industrial prop for a mobile game.",
        asset_kind="prop",
        behavior="static",
        anatomy_family=None,
        height_m=0.72,
        dimensions_m=[1.2, 0.72, 0.8],
        silhouette=["sealed rectangular body", "reinforced outer frame"],
        materials=["painted timber", "dark steel"],
        components=[
            StudioComponent(
                component_id="crate_body",
                role="body",
                connection="single continuous rigid prop",
                motion="none",
                description="Sealed rectangular cargo body.",
                visual_requirements=["recessed handle on each side"],
            )
        ],
        locked_features=["reinforced corners", "two recessed handles", "sealed body"],
        negative_constraints=["no open lid", "no loose surrounding objects"],
        gameplay_readability=["strong silhouette at mobile camera distance"],
    )
    store.save(run)

    root = store.run_root(run_id) / "tutorial"
    contract = root / "brief.json"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(run.spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    store.evidence(
        run,
        "D0",
        contract,
        evidence_id="tutorial-brief",
        label="Compiled tutorial brief",
        media_type="application/json",
        metrics={"synthetic": True, "iteration": 1},
    )
    first = _preview(root / "concept-a.png", accent="#c9683e", label="Concept A")
    second = _preview(root / "concept-b.png", accent="#4d8097", label="Concept B")
    store.evidence(
        run,
        "D1",
        first,
        evidence_id="tutorial-concept-a",
        label="Warm steel concept",
        media_type="image/png",
        metrics={"synthetic": True, "selectable": True, "iteration": 1, "quality_score": 0.91},
    )
    store.evidence(
        run,
        "D1",
        second,
        evidence_id="tutorial-concept-b",
        label="Cool steel concept",
        media_type="image/png",
        metrics={"synthetic": True, "selectable": True, "iteration": 1, "quality_score": 0.86},
    )
    geometry = _mesh(root / "geometry.glb")
    store.evidence(
        run,
        "D2",
        geometry,
        evidence_id="tutorial-geometry",
        label="Inspectable tutorial geometry",
        media_type="model/gltf-binary",
        metrics={"synthetic": True, "vertices": 8, "faces": 12, "watertight": True},
    )
    cleaned = _mesh(root / "cleaned.glb")
    store.evidence(
        run,
        "D3",
        cleaned,
        evidence_id="tutorial-cleaned",
        label="Cleaned tutorial mesh",
        media_type="model/gltf-binary",
        metrics={"synthetic": True, "connected_components": 1, "watertight": True},
    )
    surface = _preview(root / "surface.png", accent="#c9683e", label="Surface review")
    store.evidence(
        run,
        "D8",
        surface,
        evidence_id="tutorial-surface",
        label="Surface review",
        media_type="image/png",
        metrics={"synthetic": True, "selectable": True, "iteration": 1},
    )
    delivery = _preview(root / "delivery.png", accent="#d9a746", label="Delivery view")
    store.evidence(
        run,
        "D9",
        delivery,
        evidence_id="tutorial-delivery",
        label="Delivery render",
        media_type="image/png",
        metrics={"synthetic": True, "iteration": 1},
    )
    final_mesh = _mesh(root / "cargo-crate-final.glb")
    store.evidence(
        run,
        "D10",
        final_mesh,
        evidence_id="tutorial-final-glb",
        label="Final tutorial GLB",
        media_type="model/gltf-binary",
        metrics={"synthetic": True, "selectable": True, "iteration": 1},
    )
    report = root / "provenance.json"
    report.write_text(
        json.dumps(
            {
                "tutorial": True,
                "qualification_evidence": False,
                "message": "Deterministic sample data for learning the Studio interface.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    store.evidence(
        run,
        "D10",
        report,
        evidence_id="tutorial-provenance",
        label="Tutorial provenance report",
        media_type="application/json",
        metrics={"synthetic": True, "iteration": 1},
    )

    run.stage("D1").qwen_reviews.append(
        StudioQwenReview(
            review_id="tutorial-review-1",
            stage_id="D1",
            iteration=1,
            summary="Concept A best preserves the three locked crate features.",
            strengths=["clear silhouette", "both handles remain visible"],
            issues=[],
            candidate_ranking=["tutorial-concept-a", "tutorial-concept-b"],
            recommended_evidence_id="tutorial-concept-a",
            confidence=0.93,
            hard_requirements_satisfied=True,
        )
    )
    applicable = {"D0", "D1", "D2", "D3", "D8", "D9", "D10"}
    for stage in run.stages:
        stage.iteration = max(stage.iteration, 1)
        stage.finished_at = run.updated_at
        stage.progress = 1
        if stage.stage_id in applicable:
            stage.state = "approved"
            stage.message = "Completed with deterministic tutorial data."
        else:
            stage.applicable = False
            stage.state = "skipped"
            stage.message = "Not applicable to this static tutorial prop."
        if stage.gate_required and stage.stage_id in applicable:
            hashes = {item.evidence_id: item.sha256 for item in stage.evidence}
            selected = next(
                (item.evidence_id for item in stage.evidence if item.metrics.get("selectable")),
                None,
            )
            stage.human_decisions.append(
                StudioHumanDecision(
                    decision_id=f"tutorial-{stage.stage_id.lower()}-approval",
                    decision="approve",
                    comment="Pre-recorded tutorial decision. This is not live qualification evidence.",
                    selected_evidence_id=selected,
                    evidence_hashes=hashes,
                )
            )
    run.current_stage = "D10"
    run.state = "completed"
    store.event(run, "tutorial_created", {"synthetic": True, "qualification_evidence": False})
    return run
