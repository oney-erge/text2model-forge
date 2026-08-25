"""Headless access to Text2Model Studio runs: list, show, and record gate
decisions without the browser control plane.

This shares StudioStore and StudioCoordinator with text2model_forge.studio_web, so a
decision recorded here advances the exact same persisted run a browser
session would see on its next reload. StudioStore serializes workspace writes
across processes and rejects stale revisions, so the CLI and browser cannot
silently clobber one another.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .studio_pipeline import StudioCoordinator
from .studio_store import StudioStore


def list_runs(store: StudioStore, *, include_archived: bool = False) -> list[dict[str, Any]]:
    """One summary row per run, newest-updated first -- the same ordering
    StudioStore.list() and the dashboard already use. Archived runs (see
    StudioRun.archived) are excluded by default, matching the dashboard."""
    return [
        {
            "run_id": run.run_id,
            "title": run.title,
            "state": run.state,
            "current_stage": run.current_stage,
            "profile": run.profile,
            "archived": run.archived,
            "updated_at": run.updated_at.isoformat(),
        }
        for run in store.list()
        if include_archived or not run.archived
    ]


def show_run(store: StudioStore, run_id: str) -> dict[str, Any]:
    """A run's full persisted state -- the same document /api/run/<id> serves."""
    return json.loads(store.load(run_id).model_dump_json())


def decide(
    store: StudioStore,
    run_id: str,
    stage_id: str,
    decision: str,
    comment: str,
    selected_evidence_id: str | None,
    *,
    overrides: dict[str, Any] | None = None,
    target_stage_id: str | None = None,
    assisted_by_review_id: str | None = None,
    resume: bool = True,
    coordinator_factory: Callable[[StudioStore], StudioCoordinator] | None = None,
) -> dict[str, Any]:
    """Record one human decision -- identical validation and state-machine
    effect to the web form's /run/<id>/decision route, since both call
    StudioStore.decide() -- then, unless `resume` is False, drive the
    pipeline to its next stopping point on the calling thread and return the
    run's state once it gets there.

    `resume=False` only records the decision and returns immediately,
    matching what happens today if a browser user records a decision and
    closes the tab without the coordinator ever being resubmitted -- the run
    sits at `state="running"` until something calls submit() again.
    """
    store.decide(
        run_id,
        stage_id,
        decision,
        comment,
        selected_evidence_id,
        overrides=overrides,
        target_stage_id=target_stage_id,
        assisted_by_review_id=assisted_by_review_id,
    )
    if resume:
        coordinator = (coordinator_factory or StudioCoordinator)(store)
        try:
            coordinator.run_to_stop(run_id)
        finally:
            coordinator.close()
    return show_run(store, run_id)
