"""Recovery guidance must not claim that outside work is known to have stopped."""

from pathlib import Path

import pytest

from text2model_forge.studio_store import StudioStore
from text2model_forge.studio_web import build_server


@pytest.mark.parametrize("stage_state", ["pending", "running"])
def test_startup_guidance_describes_uncertain_external_work(tmp_path: Path, stage_state: str) -> None:
    store = StudioStore(tmp_path)
    run = store.create("uncertain-v1", "A synthetic interrupted project")
    run.state = "running"
    run.current_stage = "D2"
    run.stage("D2").state = stage_state
    store.save(run)

    server = build_server(tmp_path, port=0)
    try:
        recovered = server.store.load(run.run_id)
        stage = recovered.stage("D2")
        guidance = (stage.error or "") + " " + stage.message
        assert "external workers" in guidance
        assert "other Studio" in guidance
        assert "resume is safe" not in guidance
        assert "process stopped" not in guidance
        assert server.store.recover_interrupted_runs() == []
    finally:
        server.coordinator.close()
        server.server.server_close()


def test_legacy_interruption_without_candidates_is_not_recovered_twice(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("legacy-v1", "A synthetic previous interruption")
    run.state = "failed"
    run.current_stage = "D2"
    stage = run.stage("D2")
    stage.state = "failed"
    stage.error = "The local Studio process stopped before this automatic stage completed."
    stage.message = "Use Resume to retry this stage from its persisted inputs and history."
    store.save(run)
    before = (store.run_root(run.run_id) / "run.json").read_bytes()
    assert store.recover_interrupted_runs() == []
    assert (store.run_root(run.run_id) / "run.json").read_bytes() == before
