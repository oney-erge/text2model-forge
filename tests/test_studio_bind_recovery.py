"""A Studio launch that cannot bind must not recover another server's runs."""

import socket
from pathlib import Path

import pytest

from text2model_forge import studio_web
from text2model_forge.studio_store import StudioStore
from text2model_forge.studio_web import build_server


def test_occupied_port_does_not_recover_saved_running_job(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    with socket.socket() as occupied:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        run = store.create("running-v1", "A synthetic running project")
        run.state = "running"
        run.current_stage = "D2"
        run.stage("D2").state = "running"
        store.save(run)
        root = store.run_root(run.run_id)
        before_state = (root / "run.json").read_bytes()
        before_events = (root / "events.jsonl").read_bytes()

        with pytest.raises(OSError):
            build_server(tmp_path, port=occupied.getsockname()[1])

        assert (root / "run.json").read_bytes() == before_state
        assert (root / "events.jsonl").read_bytes() == before_events


def test_a_second_studio_cannot_bind_the_same_port(tmp_path: Path) -> None:
    first = build_server(tmp_path / "first", port=0)
    try:
        with pytest.raises(OSError):
            build_server(tmp_path / "second", port=first.server.server_address[1])
    finally:
        first.coordinator.close()
        first.server.server_close()


def test_closed_server_releases_its_port(tmp_path: Path) -> None:
    first = build_server(tmp_path, port=0)
    port = first.server.server_address[1]
    first.coordinator.close()
    first.server.server_close()
    second = build_server(tmp_path, port=port)
    try:
        assert second.server.server_address[1] == port
    finally:
        second.coordinator.close()
        second.server.server_close()


@pytest.mark.parametrize("failure", ["factory", "recovery"])
def test_failed_startup_closes_the_bound_server_and_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    servers = []

    class RecordingServer(studio_web._StudioHTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            servers.append(self)

    class Coordinator:
        closed = False

        def close(self):
            self.closed = True

    coordinator = Coordinator()

    def make_coordinator(_store):
        if failure == "factory":
            raise RuntimeError("synthetic factory failure")
        return coordinator

    def fail_recovery(_store):
        raise RuntimeError("synthetic recovery failure")

    monkeypatch.setattr(studio_web, "_StudioHTTPServer", RecordingServer)
    if failure == "recovery":
        monkeypatch.setattr(StudioStore, "recover_interrupted_runs", fail_recovery)
    with pytest.raises(RuntimeError, match="synthetic"):
        build_server(tmp_path, port=0, coordinator_factory=make_coordinator)
    assert len(servers) == 1
    assert servers[0].socket.fileno() == -1
    assert coordinator.closed is (failure == "recovery")


def test_successful_start_still_recovers_interrupted_run(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    run = store.create("interrupted-v1", "A synthetic interrupted project")
    run.state = "running"
    run.current_stage = "D2"
    run.stage("D2").state = "running"
    store.save(run)

    running = build_server(tmp_path, port=0)
    try:
        assert running.recovered == [run.run_id]
        assert running.store.load(run.run_id).state == "failed"
    finally:
        running.coordinator.close()
        running.server.server_close()
