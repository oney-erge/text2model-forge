"""Application commands and queries for Studio's local HTTP and future clients.

The web handler should translate HTTP, not own project workflow. This small
boundary keeps mutations in one place while preserving StudioStore as the
source of truth for evidence, decisions, and optimistic revisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .studio_demo import create_tutorial_run
from .studio_models import StudioRun
from .studio_pipeline import StudioCoordinator
from .studio_store import StudioStore


@dataclass(frozen=True)
class StudioApplication:
    store: StudioStore
    coordinator: StudioCoordinator

    def list_projects(self, *, include_archived: bool = False) -> list[StudioRun]:
        projects = self.store.list()
        if include_archived:
            return projects
        return [project for project in projects if not project.archived]

    def get_project(self, run_id: str) -> StudioRun:
        return self.store.load(run_id)

    def create_project(
        self,
        run_id: str,
        description: str,
        overrides: dict[str, Any],
        *,
        start: bool = True,
    ) -> StudioRun:
        project = self.store.create(run_id, description, overrides)
        if start:
            self.coordinator.submit(run_id)
        return project

    def create_tutorial(self, run_id: str) -> StudioRun:
        return create_tutorial_run(self.store, run_id)

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
        project = self.store.decide(
            run_id,
            stage_id,
            decision,
            comment,
            selected_evidence_id,
            overrides=overrides,
            target_stage_id=target_stage_id,
            assisted_by_review_id=assisted_by_review_id,
        )
        self.coordinator.submit(run_id)
        return project

    def set_archived(self, run_id: str, archived: bool) -> StudioRun:
        return self.store.set_archived(run_id, archived)

    def resume(self, run_id: str) -> bool:
        return self.coordinator.submit(run_id)
