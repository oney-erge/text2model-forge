"""D1-level regression tests for the equipment-duplication defect class.

Root cause, observed on a real 8 GB run: the base SDXL render ignored its own
"this side is empty" instruction and produced a stray shield anyway; the
deferred-shield repair pass then pasted a *second*, correctly-placed shield
without ever checking whether one already existed. D1's own free-form critic
reviewed the result and wrote "has correct equipment placement" at confidence
1.0 -- a VLM asked an open-ended question about a whole scene cannot count
reliably (see spec_conformance.py's module docstring for citations).

These tests drive the real `_run_d1` through `StudioCoordinator`, exactly as
production does, with a FakeQwen subclass that also answers
`visual_presence` -- opt-in, so the other ~150 tests in this file that use
plain `FakeQwen` (which has no such method) are completely unaffected; see
`qwen_supports_vision_presence` in studio_pipeline.py.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pytest

from text2model_forge.studio_pipeline import StudioCoordinator
from text2model_forge.studio_store import StudioStore

from test_studio import DESCRIPTION, FakeComfy, FakeQwen, _drive_to_first_d1_review


class VisionCapableFakeQwen(FakeQwen):
    """Answers `visual_presence` by matching on the question's wording, so a
    test can script exactly what the shield-repair guard and the
    conformance check each "see" without needing a live model."""

    def __init__(self, *, shield_already_present: bool = False, duplicate_on_opposite_side: bool = False) -> None:
        self.shield_already_present = shield_already_present
        self.duplicate_on_opposite_side = duplicate_on_opposite_side
        self.presence_calls: list[str] = []
        self.review_calls = 0

    def review_concepts(self, *args, **kwargs):
        self.review_calls += 1
        return super().review_concepts(*args, **kwargs)

    def visual_presence(self, image_path: Path, question: str, *, max_tokens: int = 40) -> bool:
        self.presence_calls.append(question)
        if "already clearly show" in question:
            return self.shield_already_present  # the pre-repair guard question
        if "second, separate" in question:
            return self.duplicate_on_opposite_side  # the conformance duplicate question
        return True  # the conformance presence question: sword/shield both "visible"


def test_shield_repair_is_skipped_when_the_target_region_already_has_it(tmp_path: Path) -> None:
    """The 'measure before act' fix: if the base render already produced the
    deferred item at its own target socket, the repair must not run at all
    -- running it anyway is exactly what risked pasting a second one."""
    qwen = VisionCapableFakeQwen(shield_already_present=True)
    comfy = FakeComfy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        store = StudioStore(tmp_path)
        _drive_to_first_d1_review(store, "shield-guard-v1", qwen, executor, comfy=comfy)

    run = store.load("shield-guard-v1")
    candidates = [item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True]
    assert candidates, "D1 must still produce selectable candidates"
    for candidate in candidates:
        assert candidate.metrics["deferred_shield_repaired"] is False
        assert candidate.metrics["deferred_shield_repair_skipped_reason"] == "already_present_in_target_region"
    # Exactly one generate() call per candidate (the base render only): if the
    # guard had NOT skipped the repair, a second ComfyUI workflow (the shield
    # inpaint) would have been submitted for each candidate too, doubling
    # this count.
    assert len(comfy.workflows) == len(candidates)
    assert any("already clearly show" in call for call in qwen.presence_calls)


def test_shield_repair_still_runs_when_the_target_region_is_genuinely_empty(tmp_path: Path) -> None:
    """The complement: when the guard correctly reports absence, the repair
    must still run exactly as before -- this is not a regression that
    disables the repair pass, only one that makes it conditional."""
    qwen = VisionCapableFakeQwen(shield_already_present=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        store = StudioStore(tmp_path)
        _drive_to_first_d1_review(store, "shield-guard-v2", qwen, executor)

    run = store.load("shield-guard-v2")
    candidates = [item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True]
    assert candidates
    for candidate in candidates:
        assert candidate.metrics["deferred_shield_repaired"] is True
        assert candidate.metrics["deferred_shield_repair_skipped_reason"] == ""


def _candidate_evidence(run):
    """The candidate images themselves, excluding every per-candidate sidecar.

    Each candidate now writes several companion artifacts whose evidence_id is
    the candidate's own id plus a suffix (`-equipment-conformance`, `-quality`,
    the geometry-ready alpha). They all contain "-candidate-", so matching on
    that alone picks them up too. They are distinguished by carrying a `role`
    metric, which the candidate image itself does not -- filter on that rather
    than on a growing list of suffixes.

    Deliberately NOT filtered by `selectable`: a duplicated candidate is
    selectable=False by design, which is the exact thing under test here.
    """
    return [
        item
        for item in run.stage("D1").evidence
        if "-candidate-" in item.evidence_id
        and item.media_type == "image/png"
        and "role" not in item.metrics
    ]


def _run_one_d1_attempt(store: StudioStore, run_id: str, qwen, executor):
    """Drive exactly one D1 attempt and return once it has settled.

    Cannot use `_drive_to_first_d1_review` for a duplicated candidate: marking
    every candidate unselectable now trips D1's pre-existing
    `automatically_retry` branch, so the run re-enters D1 instead of reaching
    `awaiting_review`. Waiting for a gate that is deliberately never offered
    would just time out.
    """
    store.create(run_id, DESCRIPTION)
    coordinator = StudioCoordinator(
        store,
        qwen_factory=lambda run: qwen,
        comfy_factory=lambda run: FakeComfy(),
        executor=executor,
    )
    assert coordinator.submit(run_id) is True
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        run = store.load(run_id)
        if _candidate_evidence(run):
            return run
        time.sleep(0.02)
    raise AssertionError(f"D1 never produced a candidate: {store.load(run_id).state}")


def test_a_duplicated_shield_is_recorded_as_a_conformance_violation(tmp_path: Path) -> None:
    """The exact real-world case: check_equipment_conformance() detects a
    second shield on the opposite side and records it as a violation on the
    candidate's own evidence -- visible on the gate, not silently absorbed
    into an unrelated aggregate 'hard_requirements_satisfied' boolean the
    reviewer's own free-form judgement already proved unreliable for this."""
    qwen = VisionCapableFakeQwen(shield_already_present=False, duplicate_on_opposite_side=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        store = StudioStore(tmp_path)
        run = _run_one_d1_attempt(store, "shield-dup-v1", qwen, executor)

    candidates = _candidate_evidence(run)
    assert candidates
    for candidate in candidates:
        assert candidate.metrics["equipment_conformance_ok"] is False
        assert "duplicated" in candidate.metrics["equipment_conformance_violations"]
    # and a dedicated evidence artifact was written, reachable on the gate
    conformance_evidence = [
        item for item in run.stage("D1").evidence if item.evidence_id.endswith("-equipment-conformance")
    ]
    assert conformance_evidence
    assert qwen.review_calls == 0, "deterministically rejected candidates must not consume VLM context"


def test_a_duplicated_candidate_is_never_offered_as_approvable(tmp_path: Path) -> None:
    """The hard gate this whole mechanism exists for. Two layers must hold:

    the candidate is marked unselectable, and StudioStore.decide() refuses to
    approve unselectable evidence. Together they mean a duplicated concept
    cannot reach D2 by any route -- not by a human clicking it, and not by the
    critic recommending it.
    """
    qwen = VisionCapableFakeQwen(shield_already_present=False, duplicate_on_opposite_side=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        store = StudioStore(tmp_path)
        run = _run_one_d1_attempt(store, "shield-dup-blocked-v1", qwen, executor)

    candidates = _candidate_evidence(run)
    assert candidates
    assert all(item.metrics["selectable"] is False for item in candidates)

    # and the store refuses the approval even if the gate were somehow offered
    # Reload after the background worker has stopped. The earlier snapshot was
    # intentionally captured while D1 was still retrying and is stale under
    # StudioStore's optimistic concurrency contract.
    run = store.load("shield-dup-blocked-v1")
    candidates = _candidate_evidence(run)
    stage = run.stage("D1")
    stage.state = "awaiting_review"
    store.save(run)
    with pytest.raises(ValueError, match="select a production candidate"):
        store.decide("shield-dup-blocked-v1", "D1", "approve", "", candidates[0].evidence_id)


def test_a_correctly_equipped_candidate_has_no_conformance_violations(tmp_path: Path) -> None:
    qwen = VisionCapableFakeQwen(shield_already_present=False, duplicate_on_opposite_side=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        store = StudioStore(tmp_path)
        _drive_to_first_d1_review(store, "shield-ok-v1", qwen, executor)

    run = store.load("shield-ok-v1")
    candidates = _candidate_evidence(run)
    assert candidates
    for candidate in candidates:
        assert candidate.metrics["equipment_conformance_ok"] is True
        assert candidate.metrics["equipment_conformance_violations"] == ""


def test_a_correctly_equipped_candidate_can_still_be_approved(tmp_path: Path) -> None:
    """The complement of the blocked-approval test: the new gate must not
    block approval in general, only the specific case of a detected
    duplicate."""
    qwen = VisionCapableFakeQwen(shield_already_present=False, duplicate_on_opposite_side=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        store = StudioStore(tmp_path)
        _drive_to_first_d1_review(store, "shield-ok-approve-v1", qwen, executor)

    run = store.load("shield-ok-approve-v1")
    candidates = _candidate_evidence(run)
    assert candidates
    store.decide("shield-ok-approve-v1", "D1", "approve", "", candidates[0].evidence_id)
    run = store.load("shield-ok-approve-v1")
    assert run.stage("D1").state == "approved"
    assert not any(item.evidence_id.endswith("-equipment-conformance") for item in run.stage("D1").evidence)


def test_a_plain_fakeqwen_without_vision_support_is_unaffected(tmp_path: Path) -> None:
    """The defensive guard itself: a qwen provider that does not implement
    visual_presence (every other test in this suite) must reach the D1 gate
    exactly as it did before this mechanism existed -- unconditional repair,
    no conformance report, no AttributeError."""
    qwen = FakeQwen()
    assert not hasattr(qwen, "visual_presence")
    with ThreadPoolExecutor(max_workers=1) as executor:
        store = StudioStore(tmp_path)
        _drive_to_first_d1_review(store, "shield-no-vision-v1", qwen, executor)

    run = store.load("shield-no-vision-v1")
    candidates = [item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True]
    assert candidates
    for candidate in candidates:
        assert candidate.metrics["deferred_shield_repaired"] is True
        assert candidate.metrics["equipment_conformance_ok"] is True
