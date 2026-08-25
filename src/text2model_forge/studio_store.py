"""Atomic run persistence, evidence hashing, and review decision semantics."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

from .studio_models import (
    StudioEvidence,
    StudioHumanDecision,
    StudioQwenReview,
    StudioRun,
    new_studio_run,
    validate_stage_overrides,
)


# Must stay identical to StudioRun.run_id's own pattern. run_root() turns a
# run id straight into a filesystem path, and the web layer passes an
# unauthenticated URL segment into it, so a looser rule here (one that
# accepted "." or "..") would resolve outside the run directory.
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class StudioConflictError(RuntimeError):
    """A caller tried to save a run based on stale persisted state."""


class _WorkspaceMutex:
    """A re-entrant process and cross-process workspace mutex.

    Studio can be driven by the browser and the headless CLI. A plain RLock
    protects threads inside one Python process but does nothing when both
    entry points use the same workspace. This mutex keeps the fast in-process
    lock and adds one advisory file lock shared by every process.
    """

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._thread_lock = threading.RLock()
        self._local = threading.local()

    @staticmethod
    def _lock_file(handle) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock_file(handle) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def __enter__(self):
        self._thread_lock.acquire()
        depth = getattr(self._local, "depth", 0)
        if depth == 0:
            handle = self.lock_path.open("a+b")
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            try:
                self._lock_file(handle)
            except Exception:
                handle.close()
                self._thread_lock.release()
                raise
            self._local.handle = handle
        self._local.depth = depth + 1
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        depth = self._local.depth - 1
        self._local.depth = depth
        if depth == 0:
            handle = self._local.handle
            try:
                self._unlock_file(handle)
            finally:
                handle.close()
                del self._local.handle
        self._thread_lock.release()


_MUTEX_REGISTRY_LOCK = threading.Lock()
_MUTEX_REGISTRY: dict[str, _WorkspaceMutex] = {}


def _workspace_mutex(root: Path) -> _WorkspaceMutex:
    key = os.path.normcase(str(root.resolve()))
    with _MUTEX_REGISTRY_LOCK:
        mutex = _MUTEX_REGISTRY.get(key)
        if mutex is None:
            mutex = _WorkspaceMutex(root / ".workspace.lock")
            _MUTEX_REGISTRY[key] = mutex
        return mutex


class StudioStore:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / "studio"
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = _workspace_mutex(self.root)

    def run_root(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("invalid run id")
        return self.runs_root / run_id

    def create(
        self, run_id: str, description: str, overrides: dict[str, Any] | None = None
    ) -> StudioRun:
        run = new_studio_run(run_id, description, overrides)
        root = self.run_root(run_id)
        with self._lock:
            if root.exists():
                raise FileExistsError(f"studio run already exists: {run_id}")
            root.mkdir(parents=True)
            self.save(run)
            self.event(run, "run_created", {"description": description})
        return run

    def load(self, run_id: str) -> StudioRun:
        path = self.run_root(run_id) / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"unknown studio run: {run_id}")
        with self._lock:
            return StudioRun.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, run: StudioRun) -> None:
        with self._lock:
            path = self.run_root(run.run_id) / "run.json"
            if path.is_file():
                current = json.loads(path.read_text(encoding="utf-8"))
                persisted_revision = int(current.get("revision", 0))
                if persisted_revision != run.revision:
                    raise StudioConflictError(
                        f"studio run {run.run_id!r} changed after it was loaded "
                        f"(expected revision {run.revision}, found {persisted_revision}); "
                        "reload it and retry the action"
                    )
            elif run.revision != 0:
                raise StudioConflictError(
                    f"studio run {run.run_id!r} disappeared after it was loaded"
                )
            original_revision = run.revision
            original_updated_at = run.updated_at
            run.revision += 1
            run.updated_at = datetime.now(timezone.utc)
            temporary = path.with_suffix(".json.tmp")
            try:
                temporary.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
                # Path.replace() is atomic on POSIX but can still raise a transient
                # PermissionError on Windows if another process or thread (real-time
                # antivirus, an unlocked reader) briefly has run.json open at the
                # exact moment of the rename. This is not a real conflict -- retry
                # rather than fail the whole stage over a race that clears in
                # milliseconds; only propagate if it is still happening after that.
                attempts = 8
                for attempt in range(attempts):
                    try:
                        temporary.replace(path)
                        return
                    except PermissionError:
                        if attempt == attempts - 1:
                            raise
                        time.sleep(0.05 * (attempt + 1))
            except Exception:
                run.revision = original_revision
                run.updated_at = original_updated_at
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

    def list(self) -> list[StudioRun]:
        with self._lock:
            runs = []
            for path in self.runs_root.glob("*/run.json"):
                try:
                    runs.append(StudioRun.model_validate_json(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
            return sorted(runs, key=lambda item: item.updated_at, reverse=True)

    def recover_interrupted_runs(self) -> list[str]:
        """Release runs left in `running` by a stopped local Studio process."""
        recovered: list[str] = []
        for run in self.list():
            stage = run.stage(run.current_stage)
            interrupted_failure = (
                run.state == "failed"
                and stage.error is not None
                and "local Studio process stopped" in stage.error
            )
            if run.state != "running" and not interrupted_failure:
                continue
            before = (run.state, stage.state, stage.error, stage.message)
            if stage.state not in {"running", "failed"}:
                run.state = "failed"
                stage.error = "Studio stopped between stages; resume is safe."
                stage.message = "The prior local process stopped. Use Resume to continue from saved state."
            else:
                current_candidates = [
                    item
                    for item in stage.evidence
                    if item.metrics.get("iteration") == stage.iteration
                    and item.metrics.get("selectable") is not False
                    and item.media_type.startswith("image/")
                    and "candidate" in item.evidence_id
                ]
                if stage.gate_required and current_candidates:
                    for item in current_candidates:
                        item.metrics["selectable"] = True
                    stage.qwen_reviews.append(
                        StudioQwenReview(
                            review_id=f"{stage.stage_id.lower()}.interrupted-{stage.iteration:02d}",
                            stage_id=stage.stage_id,
                            iteration=stage.iteration,
                            summary=(
                                "The deterministic images finished, but the prior Qwen critic process was "
                                "interrupted. The images are released for human review instead of leaving the "
                                "pipeline stuck."
                            ),
                            issues=["No completed Qwen visual ranking is available for this attempt."],
                            candidate_ranking=[item.evidence_id for item in current_candidates],
                            recommended_evidence_id=None,
                            recommended_changes=[
                                "Approve only if one result satisfies the brief; otherwise reject with the exact defect."
                            ],
                            confidence=0,
                            request_human_review=True,
                        )
                    )
                    stage.state = "awaiting_review"
                    stage.progress = 1
                    stage.error = None
                    stage.message = "Generated evidence recovered after interruption and ready for your decision."
                    run.state = "awaiting_review"
                else:
                    stage.state = "failed"
                    stage.error = "The local Studio process stopped before this automatic stage completed."
                    stage.message = "Use Resume to retry this stage from its persisted inputs and history."
                    run.state = "failed"
            if (run.state, stage.state, stage.error, stage.message) == before:
                # Already recovered by an earlier start. The message this
                # recovery writes itself matches `interrupted_failure`, so
                # without this the same run is "recovered" -- and logged, and
                # given another stage_recovered event -- on every launch.
                continue
            stage.finished_at = datetime.now(timezone.utc)
            self.event(
                run,
                "stage_recovered",
                {"stage_id": stage.stage_id, "new_state": stage.state, "iteration": stage.iteration},
            )
            recovered.append(run.run_id)
        return recovered

    def event(self, run: StudioRun, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            path = self.run_root(run.run_id) / "run.json"
            if path.is_file():
                current = json.loads(path.read_text(encoding="utf-8"))
                persisted_revision = int(current.get("revision", 0))
                if persisted_revision != run.revision:
                    raise StudioConflictError(
                        f"studio run {run.run_id!r} changed before event {event_type!r} "
                        "could be recorded; reload it and retry the action"
                    )
            run.event_count += 1
            record = {
                "sequence": run.event_count,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "stage_id": run.current_stage,
                "payload": payload,
            }
            events_path = self.run_root(run.run_id) / "events.jsonl"
            original_size = events_path.stat().st_size if events_path.is_file() else 0
            try:
                with events_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                self.save(run)
            except Exception:
                run.event_count -= 1
                try:
                    with events_path.open("r+b") as stream:
                        stream.truncate(original_size)
                except OSError:
                    pass
                raise

    def read_events(self, run_id: str) -> list[dict[str, Any]]:
        path = self.run_root(run_id) / "events.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def evidence(
        self,
        run: StudioRun,
        stage_id: str,
        path: Path,
        *,
        evidence_id: str,
        label: str,
        media_type: str,
        metrics: dict[str, float | int | bool | str | None] | None = None,
    ) -> StudioEvidence:
        with self._lock:
            resolved = path.resolve()
            root = self.run_root(run.run_id).resolve()
            if root not in resolved.parents or not resolved.is_file():
                raise ValueError("evidence must be a file inside the studio run")
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            item = StudioEvidence(
                evidence_id=evidence_id,
                label=label,
                relative_path=resolved.relative_to(root).as_posix(),
                media_type=media_type,
                sha256=digest,
                metrics=metrics or {},
            )
            stage = run.stage(stage_id)
            stage.evidence = [
                existing for existing in stage.evidence if existing.evidence_id != evidence_id
            ]
            stage.evidence.append(item)
            self.save(run)
            return item

    _DECISIONS = {"approve", "reject", "retry", "edit", "skip", "rollback"}

    @staticmethod
    def _invalidate_from(run: StudioRun, index: int, reason: str) -> None:
        """Reset every stage from `index` onward to pending, preserving each
        stage's own human_decisions (an append-only audit trail) but
        discarding evidence, reviews, and any stale pending_overrides built
        on what is now invalidated work.

        A stage D0's compiled asset contract ruled out (`applicable` is
        False -- a static prop has no rig, a material has no geometry) is
        the one exception: it keeps its skipped state and its reason.
        Invalidating work that happens to sit after it says nothing about
        whether the asset needs it, and reopening it here would send a
        static prop through the skeleton, rig, skinning, and motion stages
        the contract already excluded. Only re-running D0 -- which
        recompiles the spec and recomputes applicability -- can change that.
        """
        for downstream in run.stages[index:]:
            downstream.evidence = []
            downstream.qwen_reviews = []
            downstream.pending_overrides = {}
            downstream.error = None
            downstream.progress_phase = "waiting"
            downstream.progress_current = 0
            downstream.progress_total = 0
            downstream.progress_unit = ""
            downstream.gpu_used_gb = None
            downstream.gpu_free_gb = None
            downstream.gpu_total_gb = None
            downstream.retry_attempt = 0
            downstream.retry_reason = ""
            if not downstream.applicable:
                downstream.state = "skipped"
                downstream.progress = 1
                continue
            downstream.state = "pending"
            downstream.progress = 0
            downstream.message = reason

    def decide(
        self,
        run_id: str,
        stage_id: str,
        decision: str,
        comment: str,
        selected_evidence_id: str | None,
        *,
        overrides: dict[str, Any] | None = None,
        target_stage_id: str | None = None,
        assisted_by_review_id: str | None = None,
    ) -> StudioRun:
        """Record one human decision at a stage gate.

        approve: pick a candidate and unlock the next stage.
        reject: comment required; Qwen's next attempt sees it.
        retry: reroll the same stage, no comment required, no quality
            judgement implied. `overrides` (if given) become the stage's
            pending_overrides for its next attempt.
        edit: like reject, but the correction is concrete: comment,
            `overrides`, or both are required.
        skip: mark this stage not applicable; comment required as the
            reason; does not invalidate anything downstream.
        rollback: reopen an earlier stage named by `target_stage_id` and
            invalidate everything from there forward, including this stage.
        """
        if decision not in self._DECISIONS:
            raise ValueError(f"decision must be one of {sorted(self._DECISIONS)}")
        validate_stage_overrides(overrides)
        if decision == "reject" and not comment.strip():
            raise ValueError("a rejection comment is required so Qwen knows what to improve")
        if decision == "edit" and not comment.strip() and not overrides:
            raise ValueError("an edit needs a comment, override values, or both")
        if decision == "skip" and not comment.strip():
            raise ValueError("a skip reason is required so the record explains why the stage was bypassed")
        if decision == "rollback" and not target_stage_id:
            raise ValueError("rollback requires target_stage_id")
        with self._lock:
            run = self.load(run_id)
            stage = run.stage(stage_id)
            if not stage.gate_required or stage.state != "awaiting_review":
                raise ValueError(f"{stage_id} is not waiting for a human decision")
            evidence_ids = {item.evidence_id for item in stage.evidence}
            if selected_evidence_id and selected_evidence_id not in evidence_ids:
                raise ValueError("selected evidence does not belong to this gate")
            if assisted_by_review_id:
                assisted_review = next(
                    (
                        review
                        for review in stage.qwen_reviews
                        if review.review_id == assisted_by_review_id
                        and review.stage_id == stage.stage_id
                        and review.iteration == stage.iteration
                    ),
                    None,
                )
                if assisted_review is None:
                    raise ValueError("assisted review must belong to the current stage attempt")
                recommended_item = next(
                    (
                        item
                        for item in stage.evidence
                        if item.evidence_id == assisted_review.recommended_evidence_id
                        and item.metrics.get("selectable") is True
                    ),
                    None,
                )
                assisted_decision = (
                    "approve"
                    if assisted_review.hard_requirements_satisfied and recommended_item is not None
                    else "reject"
                )
                if decision != assisted_decision:
                    raise ValueError(
                        f"review {assisted_by_review_id} recommends {assisted_decision}, not {decision}"
                    )
                if assisted_decision == "approve" and selected_evidence_id != recommended_item.evidence_id:
                    raise ValueError("an assisted approval must select the review's recommended candidate")
            if decision == "approve" and selected_evidence_id:
                selected = next(item for item in stage.evidence if item.evidence_id == selected_evidence_id)
                if selected.metrics.get("selectable") is not True:
                    raise ValueError("select a production candidate, not a comparison or report file")
            if decision == "approve" and len(stage.evidence) > 1 and not selected_evidence_id:
                recommended = stage.qwen_reviews[-1].recommended_evidence_id if stage.qwen_reviews else None
                recommended_item = next(
                    (item for item in stage.evidence if item.evidence_id == recommended), None
                )
                if recommended_item is None or recommended_item.metrics.get("selectable") is not True:
                    raise ValueError("select one candidate before approving")
                selected_evidence_id = recommended
            target_index: int | None = None
            if decision == "rollback":
                stage_ids = [item.stage_id for item in run.stages]
                if target_stage_id not in stage_ids:
                    raise ValueError(f"unknown stage: {target_stage_id}")
                target_index = stage_ids.index(target_stage_id)
                current_index = stage_ids.index(stage_id)
                if target_index >= current_index:
                    raise ValueError(
                        "rollback target must be an earlier stage than the stage the decision is recorded against"
                    )
                if run.stages[target_index].state not in {"approved", "skipped", "rejected", "failed"}:
                    raise ValueError(f"{target_stage_id} has no prior decision to roll back to")
            record = StudioHumanDecision(
                decision_id=f"{stage_id.lower()}.{uuid.uuid4().hex[:12]}",
                decision=decision,
                comment=comment.strip(),
                selected_evidence_id=selected_evidence_id,
                evidence_hashes={item.evidence_id: item.sha256 for item in stage.evidence},
                overrides=overrides or {},
                target_stage_id=target_stage_id if decision == "rollback" else None,
                assisted_by_review_id=assisted_by_review_id,
            )
            stage.human_decisions.append(record)
            if decision != "rollback":
                # rollback's state/message for every affected stage, including
                # this one when it falls in range, is set by _invalidate_from below.
                state_by_decision = {
                    "approve": "approved",
                    "reject": "rejected",
                    "retry": "pending",
                    "edit": "rejected",
                    "skip": "skipped",
                }
                message_by_decision = {
                    "approve": "Approved. The next deterministic stage may run.",
                    "reject": "Rejected. Qwen will use the comment and complete history for the next attempt.",
                    "retry": "Retrying with a fresh attempt.",
                    "edit": "Correction recorded. The next attempt will apply it.",
                    "skip": "Skipped: " + comment.strip(),
                }
                stage.state = state_by_decision[decision]
                stage.message = message_by_decision[decision]
            if decision in {"retry", "edit"}:
                stage.pending_overrides = overrides or {}
            run.state = "running"
            event_type_by_decision = {
                "approve": "gate_approved",
                "reject": "gate_rejected",
                "retry": "gate_retried",
                "edit": "gate_edited",
                "skip": "gate_skipped",
                "rollback": "gate_rolled_back",
            }
            self.event(run, event_type_by_decision[decision], record.model_dump(mode="json"))
            if decision in {"reject", "retry", "edit"}:
                index = next(i for i, item in enumerate(run.stages) if item.stage_id == stage_id)
                self._invalidate_from(run, index + 1, "Invalidated by an upstream " + decision + ".")
                run.current_stage = stage_id
            elif decision == "rollback":
                assert target_index is not None
                self._invalidate_from(run, target_index, f"Reopened by a rollback from {stage_id}.")
                run.current_stage = target_stage_id
            self.save(run)
            return run

    def artifact_path(self, run_id: str, relative_path: str) -> Path:
        root = self.run_root(run_id).resolve()
        target = (root / relative_path).resolve()
        if root not in target.parents or not target.is_file():
            raise FileNotFoundError("artifact not found")
        return target

    def set_archived(self, run_id: str, archived: bool) -> StudioRun:
        """Flip a run's dashboard visibility. Never touches `state` or any
        stage -- an archived run can still be resumed, decided on, or
        recovered exactly as before; it just stops appearing in the default
        list. Recorded as an event for the same reason every other action
        here is: an append-only record of who changed what, and when."""
        with self._lock:
            run = self.load(run_id)
            if run.archived == archived:
                return run
            run.archived = archived
            self.event(run, "run_archived" if archived else "run_unarchived", {})
            return run
