from __future__ import annotations

from pathlib import Path

from text2model_forge.studio_application import StudioApplication
from text2model_forge.studio_store import StudioStore


class RecordingCoordinator:
    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit(self, run_id: str) -> bool:
        self.submitted.append(run_id)
        return True


def test_application_owns_project_creation_and_submission(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    coordinator = RecordingCoordinator()
    application = StudioApplication(store, coordinator)  # type: ignore[arg-type]

    project = application.create_project(
        "boundary-project",
        "A compact static crate with reinforced corners and two recessed handles.",
        {"profile": "simple"},
    )

    assert project.run_id == "boundary-project"
    assert coordinator.submitted == ["boundary-project"]
    assert application.get_project("boundary-project").revision >= 1


def test_application_queries_hide_archived_projects_by_default(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    application = StudioApplication(store, RecordingCoordinator())  # type: ignore[arg-type]
    application.create_project(
        "archived-project",
        "A compact static crate with reinforced corners and two recessed handles.",
        {"profile": "simple"},
        start=False,
    )
    application.set_archived("archived-project", True)

    assert application.list_projects() == []
    assert [item.run_id for item in application.list_projects(include_archived=True)] == [
        "archived-project"
    ]


def test_application_creates_an_explicit_offline_tutorial(tmp_path: Path) -> None:
    application = StudioApplication(
        StudioStore(tmp_path), RecordingCoordinator()  # type: ignore[arg-type]
    )

    project = application.create_tutorial("tutorial-boundary")

    assert project.run_mode == "tutorial"
    assert project.state == "completed"
