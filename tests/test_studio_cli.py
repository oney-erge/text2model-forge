"""Tests for the headless `text2model_forge studio list/show/decide` CLI surface.

These exercise text2model_forge.studio_cli's functions directly against fakes, the
same boundary tests/test_studio_web.py uses for build_server() -- cli.py's
own argv-to-dispatch plumbing is covered separately by test_studio_cli_argv
below, which only checks that argparse wires flags to the right values
(cli.py itself is otherwise untested at the argv level, same as every other
subcommand; see README's "Known gaps").
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import pytest

from text2model_forge.cli import _parser
from text2model_forge.studio_cli import decide, list_runs, show_run
from text2model_forge.studio_pipeline import StudioCoordinator
from text2model_forge.studio_store import StudioStore

from test_studio import BlockingControlFakeComfy, DESCRIPTION, FakeComfy, FakeQwen, wait_for


def test_list_runs_reports_every_run_summary(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    store.create("run-a", DESCRIPTION)
    store.create("run-b", DESCRIPTION)
    rows = list_runs(store)
    assert {row["run_id"] for row in rows} == {"run-a", "run-b"}
    assert {row["state"] for row in rows} == {"created"}
    assert all("updated_at" in row and "current_stage" in row for row in rows)


def test_show_run_matches_the_stores_own_persisted_state(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    store.create("run-a", DESCRIPTION)
    shown = show_run(store, "run-a")
    assert shown["run_id"] == "run-a"
    assert shown["description"] == DESCRIPTION
    with pytest.raises(FileNotFoundError):
        show_run(store, "does-not-exist")


def test_decide_records_and_resumes_by_default(tmp_path: Path) -> None:
    """The default (resume=True) path drives the pipeline synchronously and
    returns only once it reaches its next stopping point -- here, D1's own
    human gate again, since FakeQwen always renders two more candidates."""
    store = StudioStore(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store, qwen_factory=lambda run: FakeQwen(), comfy_factory=lambda run: FakeComfy(), executor=executor
        )
        store.create("cli-decide-v1", DESCRIPTION)
        assert coordinator.submit("cli-decide-v1")
        run = wait_for(store, "cli-decide-v1", "awaiting_review")
        candidate = next(
            item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True
        )

        result = decide(
            store,
            "cli-decide-v1",
            "D1",
            "reject",
            "Make the shield larger.",
            candidate.evidence_id,
            coordinator_factory=lambda store: StudioCoordinator(
                store,
                qwen_factory=lambda run: FakeQwen(),
                comfy_factory=lambda run: FakeComfy(),
                executor=executor,
            ),
        )
    assert result["state"] == "awaiting_review"
    stage = next(item for item in result["stages"] if item["stage_id"] == "D1")
    assert stage["iteration"] == 2
    assert stage["human_decisions"][-1]["decision"] == "reject"


def test_decide_with_no_resume_only_records_the_decision(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as executor:
        coordinator = StudioCoordinator(
            store, qwen_factory=lambda run: FakeQwen(), comfy_factory=lambda run: FakeComfy(), executor=executor
        )
        store.create("cli-no-resume-v1", DESCRIPTION)
        assert coordinator.submit("cli-no-resume-v1")
        run = wait_for(store, "cli-no-resume-v1", "awaiting_review")
        candidate = next(
            item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True
        )

        result = decide(
            store,
            "cli-no-resume-v1",
            "D1",
            "approve",
            "Looks good.",
            candidate.evidence_id,
            resume=False,
        )
    # the decision is recorded, but nothing advanced the pipeline
    assert result["state"] == "running"
    stage = next(item for item in result["stages"] if item["stage_id"] == "D1")
    assert stage["state"] == "approved"
    assert result["current_stage"] == "D1"


def test_run_to_stop_refuses_a_concurrent_job_on_the_same_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression-shaped: run_to_stop() must not silently start a second
    _drive() over the same run while one is already active on this
    coordinator -- that would race on run.json writes from two threads."""
    store = StudioStore(tmp_path)
    store.create("cli-concurrent-v1", DESCRIPTION)
    stage = store.load("cli-concurrent-v1").stage("D0")
    comfy = BlockingControlFakeComfy()
    with ThreadPoolExecutor(max_workers=2) as executor:
        coordinator = StudioCoordinator(
            store,
            qwen_factory=lambda run: FakeQwen(),
            comfy_factory=lambda run: comfy,
            executor=executor,
        )
        assert coordinator.submit("cli-concurrent-v1")
        with pytest.raises(RuntimeError, match="already running"):
            coordinator.run_to_stop("cli-concurrent-v1")

        original_event = store.event
        inserted_concurrent_event = False

        def event_after_concurrent_write(run, event_type, payload):
            nonlocal inserted_concurrent_event
            if event_type == "human_stop_requested" and not inserted_concurrent_event:
                inserted_concurrent_event = True
                latest = store.load("cli-concurrent-v1")
                original_event(latest, "concurrent_test_event", {})
            return original_event(run, event_type, payload)

        monkeypatch.setattr(store, "event", event_after_concurrent_write)
        coordinator.stop("cli-concurrent-v1")
        assert inserted_concurrent_event
        assert any(event["event_type"] == "human_stop_requested" for event in store.read_events("cli-concurrent-v1"))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and coordinator.busy("cli-concurrent-v1"):
            time.sleep(0.02)


def test_argv_parses_every_studio_decide_flag(tmp_path: Path) -> None:
    """cli.py's own argparse wiring: no-subcommand keeps launching the
    browser server exactly as before, and each new subcommand's flags reach
    the right namespace attributes."""
    parser = _parser()
    serve_args = parser.parse_args(
        [
            "studio",
            "--workspace",
            str(tmp_path),
            "--open-browser",
            "--allow-non-loopback",
        ]
    )
    assert serve_args.command == "studio"
    assert getattr(serve_args, "studio_command", None) is None
    assert serve_args.open_browser is True
    assert serve_args.allow_non_loopback is True

    list_args = parser.parse_args(["studio", "list", "--workspace", str(tmp_path)])
    assert list_args.studio_command == "list"

    show_args = parser.parse_args(["studio", "show", "--workspace", str(tmp_path), "--run-id", "r1"])
    assert show_args.studio_command == "show"
    assert show_args.run_id == "r1"

    decide_args = parser.parse_args(
        [
            "studio",
            "decide",
            "--workspace",
            str(tmp_path),
            "--run-id",
            "r1",
            "--stage-id",
            "D1",
            "--decision",
            "edit",
            "--comment",
            "Widen the shield.",
            "--selected-evidence-id",
            "cand-1",
            "--overrides",
            '{"seed": 7}',
            "--assisted-by-review-id",
            "review-1",
            "--no-resume",
        ]
    )
    assert decide_args.studio_command == "decide"
    assert decide_args.decision == "edit"
    assert decide_args.comment == "Widen the shield."
    assert decide_args.selected_evidence_id == "cand-1"
    assert decide_args.overrides == '{"seed": 7}'
    assert decide_args.assisted_by_review_id == "review-1"
    assert decide_args.no_resume is True


def test_list_runs_excludes_archived_by_default(tmp_path: Path) -> None:
    store = StudioStore(tmp_path)
    store.create("archived-run", DESCRIPTION)
    store.create("visible-run", DESCRIPTION)
    store.set_archived("archived-run", True)

    assert {row["run_id"] for row in list_runs(store)} == {"visible-run"}
    all_rows = list_runs(store, include_archived=True)
    assert {row["run_id"] for row in all_rows} == {"archived-run", "visible-run"}
    archived_row = next(row for row in all_rows if row["run_id"] == "archived-run")
    assert archived_row["archived"] is True


def test_argv_parses_the_include_archived_flag() -> None:
    parser = _parser()
    args = parser.parse_args(["studio", "list", "--workspace", "X", "--include-archived"])
    assert args.include_archived is True
    args = parser.parse_args(["studio", "list", "--workspace", "X"])
    assert args.include_archived is False
