"""Verify recovery guidance through the actual loopback Studio page."""

import threading
import urllib.request
from pathlib import Path

import pytest

from text2model_forge.studio_store import StudioStore
from text2model_forge.studio_web import build_server


@pytest.mark.parametrize("legacy", [None, "running", "between"])
def test_recovery_page_warns_about_external_work(tmp_path: Path, legacy: str | None) -> None:
    store = StudioStore(tmp_path)
    run = store.create("guidance-v1", "A synthetic interrupted project")
    run.current_stage = "D2"
    stage = run.stage("D2")
    if legacy == "running":
        run.state = "failed"
        stage.state = "failed"
        stage.error = "The local Studio process stopped before this automatic stage completed."
        stage.message = "Use Resume to retry this stage from its persisted inputs and history."
    elif legacy == "between":
        run.state = "failed"
        stage.state = "pending"
        stage.error = "Studio stopped between stages; resume is safe."
        stage.message = "The prior local process stopped. Use Resume to continue from saved state."
    else:
        run.state = "running"
        stage.state = "running"
    store.save(run)

    studio = build_server(tmp_path, port=0)
    thread = threading.Thread(target=studio.server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(studio.url + "/run/guidance-v1", timeout=5) as response:
            page = response.read().decode("utf-8")
        assert "other Studio instances and external workers" in page
        assert "resume is safe" not in page
        assert "The local Studio process stopped" not in page
    finally:
        studio.server.shutdown()
        studio.server.server_close()
        studio.coordinator.close()
        thread.join(timeout=5)
