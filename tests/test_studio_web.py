"""HTTP-level tests for the Studio browser control plane.

These drive a real loopback ThreadingHTTPServer rather than calling handler
methods directly, so routing, CSRF enforcement, form parsing, redirects, and
error rendering are all exercised as a browser would exercise them. Before
this file src/text2model_forge/studio_web.py had zero test coverage.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
import html
import io
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image
import pytest

from text2model_forge.studio_pipeline import StudioCoordinator
from text2model_forge.studio_web import _run_progress, build_server

from test_studio import (
    CHAIR_DESCRIPTION,
    DESCRIPTION,
    BlockingControlFakeComfy,
    ChairQwen,
    FakeComfy,
    FakeQwen,
    FakeScriptRunner,
    FakeWorkerExecutor,
    _approve_gate,
    _machine_config_with_blender,
    _wait_for_stage_settled,
    wait_for,
)


def test_non_loopback_bind_requires_an_explicit_container_opt_in(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow_non_loopback"):
        build_server(tmp_path, host="0.0.0.0", port=0)

    running = build_server(tmp_path, host="0.0.0.0", port=0, allow_non_loopback=True)
    try:
        assert running.server.server_address[0] == "0.0.0.0"
    finally:
        running.coordinator.close()
        running.server.server_close()


@pytest.fixture
def studio(tmp_path: Path):
    """A running Studio server backed entirely by fakes, on an ephemeral port.

    The coordinator is built FROM the server's own store via the factory, not
    handed in pre-built on a store of its own: two StudioStore instances over
    the same directory hold independent locks, so the coordinator thread and
    the request handlers would race on run.json writes and list() could
    intermittently observe a run mid-replace.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = build_server(
            tmp_path,
            port=0,
            coordinator_factory=lambda store: StudioCoordinator(
                store,
                qwen_factory=lambda run: FakeQwen(),
                comfy_factory=lambda run: FakeComfy(),
                worker_executor=FakeWorkerExecutor(),
                executor=executor,
            ),
        )
        thread = threading.Thread(target=running.server.serve_forever, daemon=True)
        thread.start()
        try:
            yield running
        finally:
            running.server.shutdown()
            running.server.server_close()
            thread.join(timeout=5)


def _get(studio, path: str):
    try:
        with urllib.request.urlopen(studio.url + path, timeout=10) as response:
            return response.status, response.read().decode("utf-8"), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8"), dict(error.headers)


def _only_run_id(studio, timeout: float = 5) -> str:
    """The single run this test created. Polls rather than indexing list()[0]
    directly: store.list() skips any run.json it cannot parse, so indexing it
    the instant after a POST can raise IndexError if the coordinator thread
    happens to be mid-write."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runs = studio.store.list()
        if runs:
            return runs[0].run_id
        time.sleep(0.02)
    raise AssertionError("no run was created")


def _post(studio, path: str, fields: dict[str, str], *, follow: bool = False):
    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(studio.url + path, data=data, method="POST")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def _post_multipart(
    studio, path: str, fields: dict[str, str], *, file_field: str, filename: str, file_bytes: bytes
):
    boundary = "----WebTestBoundary7f3c9a"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    request = urllib.request.Request(
        studio.url + path,
        data=b"".join(parts),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def test_dashboard_and_new_asset_pages_render(studio):
    status, body, headers = _get(studio, "/")
    assert status == HTTPStatus.OK
    assert "Asset production runs" in body
    assert 'href="/golden"' in body

    status, body, _ = _get(studio, "/golden")
    assert status == HTTPStatus.OK
    assert "Live 8 GB static-prop qualification" in body
    assert "0/10 attempted" in body
    assert body.count("Start this case") == 10

    status, body, _ = _get(studio, "/new")
    assert status == HTTPStatus.OK
    assert "Describe one original asset" in body
    # The form carries the server's CSRF token and the profile selector.
    assert studio.csrf in body
    assert 'name=profile' in body
    assert "advanced" in body


def test_first_run_dashboard_offers_an_offline_walkthrough(studio):
    status, body, _ = _get(studio, "/")

    assert status == HTTPStatus.OK
    assert body.count("Open offline walkthrough") == 1
    assert 'method=post action=/demo' in body
    assert "Configure and create" in body
    assert "Check this computer" in body

    status, _ = _post(studio, "/demo", {"csrf": studio.csrf})
    assert status == HTTPStatus.SEE_OTHER
    project = studio.store.list()[0]
    assert project.run_mode == "tutorial"
    assert project.state == "completed"

    status, body, _ = _get(studio, f"/run/{project.run_id}")
    assert status == HTTPStatus.OK
    assert "Offline tutorial" in body
    assert "Download final GLB" in body
    assert "not qualification evidence" in body


def test_app_shell_and_creation_form_have_responsive_accessible_structure(studio):
    _, body, _ = _get(studio, "/new")

    assert '<html lang="en">' in body
    assert 'class="skip-link"' in body
    assert body.count("<h1>") == 1
    assert 'aria-label="Primary"' in body
    assert "overflow-x: hidden" in body
    assert "@media (max-width: 820px)" in body
    for control_id in (
        "description",
        "intended-use",
        "must-have",
        "profile",
        "concept-backend",
        "checkpoint",
        "review-model",
        "spec-strategy",
        "concept-steps",
        "concept-cfg",
        "concept-candidates",
        "device-policy",
    ):
        assert f"for={control_id}" in body
        assert f"id={control_id}" in body


def test_project_api_and_optional_brief_context(studio):
    status, _ = _post(
        studio,
        "/runs",
        {
            "csrf": studio.csrf,
            "description": DESCRIPTION,
            "profile": "simple",
            "intended_use": "Game-ready asset",
            "must_have_features": "Separate handle and clean silhouette",
        },
    )
    assert status == HTTPStatus.SEE_OTHER
    project = studio.store.load(_only_run_id(studio))
    assert project.description == DESCRIPTION
    assert project.intended_use == "Game-ready asset"
    assert project.must_have_features == "Separate handle and clean silhouette"
    assert "Intended use: Game-ready asset" in project.compilation_brief()

    status, body, _ = _get(studio, "/api/v1/projects")
    assert status == HTTPStatus.OK
    listing = json.loads(body)
    assert listing[0]["run_id"] == project.run_id
    assert listing[0]["run_id"] == project.run_id

    status, body, _ = _get(studio, f"/api/v1/projects/{project.run_id}")
    assert status == HTTPStatus.OK
    assert json.loads(body)["revision"] >= 1

    status, body, _ = _get(studio, f"/api/v1/projects/{project.run_id}/events")
    assert status == HTTPStatus.OK
    assert any(item["event_type"] == "run_created" for item in json.loads(body))


def test_review_workspace_focuses_evidence_and_the_human_gate(studio):
    _post(
        studio,
        "/runs",
        {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"},
    )
    project = wait_for(studio.store, _only_run_id(studio), "awaiting_review")

    status, body, _ = _get(studio, f"/run/{project.run_id}")

    assert status == HTTPStatus.OK
    assert 'class=run-workspace' in body
    assert 'class=phase-list' in body
    assert 'class="evidence-grid"' in body
    assert 'id=human-decision' in body
    assert 'form=human-decision' in body
    assert 'for=decision-comment' in body
    assert 'for=decision-overrides' in body
    assert "Request revision" in body
    assert "More decision options" in body
    assert "Technical stages and project controls" in body


def test_new_asset_page_exposes_typed_text_to_2d_and_reviewer_choices(studio):
    status, body, _ = _get(studio, "/new")
    assert status == HTTPStatus.OK
    assert 'name=concept_backend' in body
    assert "Qwen Image 2512" in body
    assert "SDXL checkpoint" in body
    assert 'name=checkpoint' in body
    assert 'name=model' in body
    assert 'name=spec_strategy' in body
    assert 'name=concept_steps' in body
    assert 'name=concept_cfg' in body
    assert "/static/studio.js" in body


def test_new_asset_form_hides_advanced_options_for_the_simple_profile(
    studio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The custom-checkpoint/device-policy panel used to render <details
    ... open> unconditionally -- every field visible to every user
    regardless of the profile they picked. It should stay closed by default
    for "simple" and open for anything else; the user simple-recommended
    machines had that panel show, but it must still be reachable, just not
    shoved in front of a simple-profile user by default."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "text2model_forge.studio_web.recommend_stack",
        lambda hardware: SimpleNamespace(profile="simple"),
    )
    _, body, _ = _get(studio, "/new")
    assert "<details class=options>" in body
    assert "<details class=options open>" not in body

    monkeypatch.setattr(
        "text2model_forge.studio_web.recommend_stack",
        lambda hardware: SimpleNamespace(profile="advanced"),
    )
    _, body, _ = _get(studio, "/new")
    assert "<details class=options open>" in body


def test_new_asset_form_explains_advanced_fields_with_hints_not_permanent_paragraphs(
    studio,
) -> None:
    """The options panel used to carry a standing <p class=muted> paragraph
    of explanation under multiple fields, all visible at once regardless of
    whether anyone needed it. Each explanation now lives in a hover/focus
    hint attached to its field's label instead."""
    _, body, _ = _get(studio, "/new")
    assert body.count("class=hint") >= 7
    assert "Which model renders D1 concept images" in body
    assert "data-backend-note" not in body


def test_creating_a_run_records_text_to_2d_and_reviewer_overrides(studio):
    status, _ = _post(
        studio,
        "/runs",
        {
            "csrf": studio.csrf,
            "description": DESCRIPTION,
            "profile": "simple",
            "concept_backend": "sdxl",
            "checkpoint": "my-installed-sdxl.safetensors",
            "model": "qwen3-vl:8b-instruct",
            "spec_strategy": "chunked",
            "concept_steps": "24",
            "concept_cfg": "5.5",
        },
    )
    assert status in {HTTPStatus.FOUND, HTTPStatus.SEE_OTHER}
    run = studio.store.list()[0]
    assert run.concept_backend == "sdxl"
    assert run.checkpoint == "my-installed-sdxl.safetensors"
    assert run.model == "qwen3-vl:8b-instruct"
    assert run.spec_strategy == "chunked"
    assert run.concept_steps == 24
    assert run.concept_cfg == 5.5


def test_unknown_text_to_2d_backend_is_rejected_before_a_run_is_created(studio):
    status, body = _post(
        studio,
        "/runs",
        {
            "csrf": studio.csrf,
            "description": DESCRIPTION,
            "profile": "simple",
            "concept_backend": "mystery-service",
        },
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "unknown text-to-2D backend" in body
    assert studio.store.list() == []


def test_security_headers_are_set_on_every_page(studio):
    _, _, headers = _get(studio, "/")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"


def test_unknown_path_is_a_clean_404_not_a_traceback(studio):
    status, body = _post(studio, "/definitely-not-a-route", {"csrf": studio.csrf})
    assert status == HTTPStatus.NOT_FOUND
    assert "Traceback" not in body


def test_post_without_a_valid_csrf_token_is_refused(studio):
    status, body = _post(
        studio,
        "/runs",
        {"csrf": "not-the-real-token", "description": "x" * 40, "profile": "simple"},
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "invalid form token" in body
    # and no run was created
    assert studio.store.list() == []


def test_creating_a_run_requires_a_substantial_description(studio):
    status, body = _post(
        studio, "/runs", {"csrf": studio.csrf, "description": "too short", "profile": "simple"}
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "at least 20 characters" in body
    assert studio.store.list() == []


def test_creating_a_run_records_the_selected_profile(studio):
    status, _ = _post(
        studio,
        "/runs",
        {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "advanced"},
    )
    assert status in {HTTPStatus.FOUND, HTTPStatus.SEE_OTHER}
    runs = studio.store.list()
    assert len(runs) == 1
    assert runs[0].profile == "advanced"
    # advanced.toml pins quality=high, which resolves to concept_steps=45
    assert runs[0].concept_steps == 45


def test_malformed_override_json_is_explained_not_dumped_as_a_traceback(studio):
    """The override guard added to the decision route is verified here for
    real rather than only by reading it."""
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    candidate = next(
        item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True
    )

    status, body = _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "retry",
            "comment": "",
            "selected_evidence_id": candidate.evidence_id,
            "overrides": "{not valid json",
        },
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "must be valid JSON" in body
    assert "Traceback" not in body

    status, body = _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "retry",
            "comment": "",
            "selected_evidence_id": candidate.evidence_id,
            "overrides": "[1, 2, 3]",
        },
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "must be a JSON object" in body

    # An out-of-range value is rejected by the shared validator, at the gate.
    status, body = _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "retry",
            "comment": "",
            "selected_evidence_id": candidate.evidence_id,
            "overrides": '{"seed": -5}',
        },
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "non-negative whole number" in body

    # None of the three rejected attempts changed the stage.
    assert studio.store.load(run.run_id).stage("D1").state == "awaiting_review"


def test_approving_a_gate_through_the_web_form_advances_the_run(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    candidate = next(
        item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True
    )
    status, _ = _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "approve",
            "comment": "Looks good.",
            "selected_evidence_id": candidate.evidence_id,
        },
    )
    assert status in {HTTPStatus.FOUND, HTTPStatus.SEE_OTHER}
    decided = studio.store.load(run.run_id)
    assert decided.stage("D1").state == "approved"
    assert decided.stage("D1").human_decisions[-1].comment == "Looks good."


def test_ai_recommendation_requires_confirmation_and_records_its_review_id(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    stage = run.stage("D1")
    review = stage.qwen_reviews[-1]
    candidate = next(item for item in stage.evidence if item.evidence_id == review.recommended_evidence_id)

    status, body, _ = _get(studio, f"/run/{run.run_id}")
    assert status == HTTPStatus.OK
    assert "AI review recommendation" in body
    assert "Confirm AI recommendation" in body
    assert review.review_id in body
    assert studio.store.load(run.run_id).stage("D1").human_decisions == []

    status, _ = _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "approve",
            "comment": "Confirmed after inspecting the candidate.",
            "selected_evidence_id": candidate.evidence_id,
            "assisted_by_review_id": review.review_id,
        },
    )
    assert status in {HTTPStatus.FOUND, HTTPStatus.SEE_OTHER}
    record = studio.store.load(run.run_id).stage("D1").human_decisions[-1]
    assert record.assisted_by_review_id == review.review_id


def test_run_page_and_json_api_expose_the_runs_state(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")

    status, body, _ = _get(studio, f"/run/{run.run_id}")
    assert status == HTTPStatus.OK
    assert run.run_id in body

    status, body, headers = _get(studio, f"/api/run/{run.run_id}")
    assert status == HTTPStatus.OK
    assert headers["Content-Type"].startswith("application/json")
    import json as _json

    payload = _json.loads(body)
    assert payload["run_id"] == run.run_id


def test_run_and_dashboard_show_normalized_accessible_pipeline_progress(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    expected = _run_progress(run)
    assert 0 < expected < 1

    _, run_body, _ = _get(studio, f"/run/{run.run_id}")
    assert "Whole pipeline" in run_body
    assert 'id="overall-run-bar"' in run_body
    assert 'role="progressbar"' in run_body
    assert f"{expected:.0%} overall" in run_body

    _, dashboard, _ = _get(studio, "/")
    assert "Pipeline progress" in dashboard
    assert f"{expected:.0%}" in dashboard


def test_setup_options_api_returns_discovered_models_without_blocking_the_form(
    studio, monkeypatch: pytest.MonkeyPatch
):
    payload = {
        "defaults": {"concept_backend": "auto", "checkpoint": "default.safetensors"},
        "checkpoints": ["default.safetensors", "custom.safetensors"],
        "diffusion_models": ["qwen_image_2512_fp8_e4m3fn.safetensors"],
        "review_models": ["qwen3-vl:8b-instruct"],
        "qwen_image_2512_ready": True,
        "services": {
            "comfyui": {"ready": True, "detail": "HTTP 200"},
            "reviewer": {"ready": True, "detail": "HTTP 200"},
        },
    }
    monkeypatch.setattr("text2model_forge.studio_web._setup_options", lambda profile: payload)
    status, body, headers = _get(studio, "/api/setup/options?profile=simple")
    assert status == HTTPStatus.OK
    assert headers["Content-Type"].startswith("application/json")
    import json as _json

    assert _json.loads(body) == payload


def test_error_block_recognizes_network_timeouts_as_a_dependency_hint() -> None:
    """A raw urllib failure -- "URLError: <urlopen error timed out>", or on
    Windows "URLError: <urlopen error [WinError 10061] ... actively refused
    it>" -- used to render as a bare traceback with no guidance, which is
    exactly what a stage failure looks like the moment ComfyUI or the
    reviewer is not running. Regression for the missing entries in
    _DEPENDENCY_ERROR_HINTS."""
    from types import SimpleNamespace

    from text2model_forge.studio_web import _error_block

    timed_out = SimpleNamespace(error="URLError: <urlopen error timed out>")
    refused = SimpleNamespace(
        error=(
            "URLError: <urlopen error [WinError 10061] No connection could be "
            "made because the target machine actively refused it>"
        )
    )
    for stage in (timed_out, refused):
        assert 'href="/doctor"' in _error_block(stage)


def test_setup_options_reports_offline_services_in_plain_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_json_get's raw exception text ("URLError: <urlopen error timed
    out>") used to reach the New-asset page verbatim -- the exact status
    line a user sees opening /new with nothing running yet. Regression for
    the raw-exception passthrough in _setup_options."""
    from text2model_forge.studio_web import _setup_options

    def offline(url: str):
        return None, "URLError: <urlopen error timed out>"

    monkeypatch.setattr("text2model_forge.studio_web._json_get", offline)
    options = _setup_options("simple")

    for service in ("comfyui", "reviewer"):
        assert options["services"][service]["ready"] is False
        detail = options["services"][service]["detail"]
        assert "urlopen error" not in detail
        assert "/doctor" in detail


def _green_screen_png() -> bytes:
    """A fixture matching FakeComfy.generate's own: an edge-connected green
    border around a non-green center, so it is valid input to
    make_chroma_alpha() -- not just any RGB PNG. Anything else would fail
    the same deterministic gate a generated candidate has to pass, which is
    the point of the upload feature, not a test-only shortcut."""
    image = Image.new("RGB", (64, 96), (0, 220, 0))
    for y in range(16, 80):
        for x in range(12, 52):
            image.putpixel((x, y), (70, 90, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_uploading_a_d1_image_adds_it_as_a_selectable_candidate(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    assert run.current_stage == "D1"

    status, _ = _post_multipart(
        studio,
        f"/run/{run.run_id}/upload-image",
        {"csrf": studio.csrf},
        file_field="image",
        filename="my-concept.png",
        file_bytes=_green_screen_png(),
    )
    assert status == HTTPStatus.SEE_OTHER

    deadline = time.monotonic() + 8
    evidence = None
    while time.monotonic() < deadline:
        stage = studio.store.load(run.run_id).stage("D1")
        assert stage.state != "failed", f"D1 failed: {stage.error}"
        evidence = next(
            (item for item in stage.evidence if item.evidence_id.endswith("-candidate-upload-01")), None
        )
        if evidence is not None:
            break
        time.sleep(0.02)
    assert evidence is not None, "uploaded candidate evidence never appeared"
    assert evidence.metrics["selectable"] is True
    assert evidence.metrics["human_uploaded"] is True
    assert evidence.metrics["meaningful_alpha"] is True

    _, body, _ = _get(studio, f"/run/{run.run_id}")
    assert evidence.evidence_id in body


def test_uploading_an_image_without_a_green_background_is_kept_but_not_selectable(studio):
    """The upload path must not silently bypass the same gate a generated
    candidate has to pass -- a real photo or plain-background image should
    still be recorded (for the same audit trail every candidate gets) but
    marked not selectable, with the reason visible, not accepted as if it
    had passed."""
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")

    plain_image = Image.new("RGB", (64, 96), (40, 60, 90))
    buffer = io.BytesIO()
    plain_image.save(buffer, format="PNG")

    status, _ = _post_multipart(
        studio,
        f"/run/{run.run_id}/upload-image",
        {"csrf": studio.csrf},
        file_field="image",
        filename="plain-photo.png",
        file_bytes=buffer.getvalue(),
    )
    assert status == HTTPStatus.SEE_OTHER

    deadline = time.monotonic() + 8
    evidence = None
    while time.monotonic() < deadline:
        stage = studio.store.load(run.run_id).stage("D1")
        assert stage.state != "failed", f"D1 failed: {stage.error}"
        evidence = next(
            (item for item in stage.evidence if item.evidence_id.endswith("-candidate-upload-01")), None
        )
        if evidence is not None:
            break
        time.sleep(0.02)
    assert evidence is not None, "uploaded candidate evidence never appeared"
    assert evidence.metrics["selectable"] is False
    assert evidence.metrics["meaningful_alpha"] is False
    assert evidence.metrics["quality_reasons"]


def test_uploading_a_non_image_file_is_rejected_with_a_clear_error(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")

    status, body = _post_multipart(
        studio,
        f"/run/{run.run_id}/upload-image",
        {"csrf": studio.csrf},
        file_field="image",
        filename="not-an-image.txt",
        file_bytes=b"just some text, not image bytes",
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "not a readable image" in body.lower()


def test_active_runs_api_is_empty_with_no_runs_and_lists_one_awaiting_review(studio):
    status, body, headers = _get(studio, "/api/active-runs")
    assert status == HTTPStatus.OK
    assert headers["Content-Type"].startswith("application/json")
    import json as _json

    assert _json.loads(body) == []

    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")

    _, body, _ = _get(studio, "/api/active-runs")
    active = _json.loads(body)
    assert len(active) == 1
    assert active[0]["run_id"] == run.run_id
    assert active[0]["state"] == "awaiting_review"
    assert active[0]["current_stage"] == run.current_stage


def test_active_runs_script_is_served_and_every_page_wires_the_header_slot(studio):
    status, body, headers = _get(studio, "/static/active-runs.js")
    assert status == HTTPStatus.OK
    assert headers["Content-Type"].startswith("text/javascript")
    assert "initActiveRuns" in body
    assert "/api/active-runs" in body

    for path in ("/", "/new", "/doctor"):
        _, body, _ = _get(studio, path)
        assert 'id=active-runs' in body
        assert '"/static/active-runs.js"' in body


def test_favicon_serves_a_real_icon_instead_of_no_content(studio):
    """/favicon.ico used to answer 204 No Content -- a browser tab with no
    icon at all. Regression for the SVG mark now served there."""
    status, body, headers = _get(studio, "/favicon.ico")
    assert status == HTTPStatus.OK
    assert headers["Content-Type"].startswith("image/svg+xml")
    assert body.startswith("<svg")


def test_page_header_shows_the_logo_mark(studio):
    _, body, _ = _get(studio, "/")
    assert '<svg class=logo' in body


def test_artifact_route_refuses_to_escape_the_run_directory(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    escape = urllib.parse.quote("../../../../etc/passwd", safe="")
    status, body, _ = _get(studio, f"/artifact/{run.run_id}/{escape}")
    assert status == HTTPStatus.NOT_FOUND
    assert "Traceback" not in body
    assert "passwd" not in body or "artifact not found" in body


def test_doctor_page_returns_its_shell_immediately(studio):
    """The System page used to run every check before sending a single byte,
    so the nav link looked like a dead button for as long as the slowest
    probe took (~6 s with everything refused, far longer against a service
    that is listening but hung). The shell must come back promptly with a
    visible checking state; the checks themselves moved to /api/doctor."""
    start = time.monotonic()
    status, body, _ = _get(studio, "/doctor")
    elapsed = time.monotonic() - start

    assert status == HTTPStatus.OK
    assert elapsed < 1.5, f"the /doctor shell should be immediate, took {elapsed:.2f}s"
    assert "doctor-shell" in body
    assert "Checking hardware" in body
    assert '"/static/doctor.js"' in body


def test_doctor_api_reports_dependency_state_without_crashing(studio):
    """The checks themselves: ComfyUI and LocalDeploy are not running here,
    so they must be reported as down rather than raise."""
    status, body, headers = _get(studio, "/api/doctor")
    assert status == HTTPStatus.OK
    assert headers["Content-Type"].startswith("text/html")
    assert "ComfyUI" in body or "comfy" in body.lower()
    assert "Preflight" in body


def test_stage_detail_page_keeps_a_decided_stages_record_reachable(studio):
    """Regression: the run page renders only the current stage, so once a run
    advanced past D1 its concept images, Qwen reviews, and the human decision
    itself were unreachable from the browser entirely."""
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    candidate = next(
        item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True
    )
    _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "approve",
            "comment": "The shield reads at sprite scale.",
            "selected_evidence_id": candidate.evidence_id,
        },
    )
    status, body, _ = _get(studio, f"/run/{run.run_id}/stage/D1")
    assert status == HTTPStatus.OK
    # the human record, the Qwen review, and the image evidence all survive
    assert "The shield reads at sprite scale." in body
    assert "Qwen reviews" in body
    assert candidate.relative_path.rsplit("/", 1)[-1] in body
    # and the run page links to it rather than stranding it
    _, run_body, _ = _get(studio, f"/run/{run.run_id}")
    assert f"/run/{run.run_id}/stage/D1" in run_body


def test_every_stage_has_a_detail_page_and_unknown_ones_are_404(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    for stage in run.stages:
        status, body, _ = _get(studio, f"/run/{run.run_id}/stage/{stage.stage_id}")
        assert status == HTTPStatus.OK, stage.stage_id
        assert stage.label in body
    status, body, _ = _get(studio, f"/run/{run.run_id}/stage/D99")
    assert status == HTTPStatus.NOT_FOUND
    assert "Traceback" not in body


def test_dashboard_flags_the_runs_that_need_a_human_decision(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    status, body, _ = _get(studio, "/")
    assert status == HTTPStatus.OK
    assert "Needs your decision" in body
    assert "1 waiting for your decision" in body


def test_artifact_route_refuses_a_relative_run_id(studio):
    """Regression: run_root() accepted "." and "..", so this URL resolved to
    the studio root and served files from outside any run directory."""
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    status, body, _ = _get(studio, "/artifact/%2e%2e/runs")
    assert status == HTTPStatus.NOT_FOUND
    assert "invalid run id" in body


def test_an_unknown_configuration_profile_is_refused(studio):
    """resolve_settings() turns this value into a profiles/<name>.toml path."""
    status, body = _post(
        studio,
        "/runs",
        {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "../../base"},
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "unknown configuration profile" in body
    assert studio.store.list() == []


def test_evidence_is_grouped_by_attempt_so_superseded_candidates_are_labelled(studio):
    """A rejected-and-retried gate accumulates every attempt's artefacts on
    one stage; the flat grid made the live candidates indistinguishable."""
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    candidate = next(
        item for item in run.stage("D1").evidence if item.metrics.get("selectable") is True
    )
    _post(
        studio,
        f"/run/{run.run_id}/decision",
        {
            "csrf": studio.csrf,
            "stage_id": "D1",
            "decision": "reject",
            "comment": "Make the shield larger.",
            "selected_evidence_id": candidate.evidence_id,
        },
    )
    second = wait_for(studio.store, run.run_id, "awaiting_review")
    assert second.stage("D1").iteration == 2
    _, body, _ = _get(studio, f"/run/{run.run_id}")
    assert "Attempt 2" in body and "current attempt" in body
    assert "Attempt 1" in body and "superseded" in body


def test_stage_page_shows_the_duration_that_was_always_recorded_but_never_shown(studio):
    """Regression: started_at/finished_at are written on every stage
    transition -- eleven call sites in studio_pipeline.py -- but nothing ever
    read them back, so no duration was reachable anywhere in the browser."""
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run = wait_for(studio.store, _only_run_id(studio), "awaiting_review")
    approved = run.stage("D0")
    assert approved.started_at is not None and approved.finished_at is not None

    status, body, _ = _get(studio, f"/run/{run.run_id}/stage/D0")
    assert status == 200
    assert "ran " in body and "s</span>" in body

    # the run page's hero also shows the *current* stage's elapsed/duration badge
    _, run_body, _ = _get(studio, f"/run/{run.run_id}")
    assert "created " in run_body


def test_a_completed_runs_final_evidence_is_reachable_from_its_stage_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished asset's delivery board and runtime manifest are ordinary
    D10 evidence -- StudioStore.evidence() writes them and /artifact/ already
    serves any evidence file, so the D10 stage page (added for every stage in
    this same change) is already 'download the finished asset' with no extra
    plumbing required. This locks that in as a driven check rather than an
    inference from reading the code.

    Builds its own server (rather than the shared `studio` fixture) because
    the chair chain needs ChairQwen and a FakeScriptRunner that fixture does
    not wire up.
    """
    monkeypatch.setattr(
        "text2model_forge.studio_pipeline.load_local_config",
        lambda *a, **k: _machine_config_with_blender(tmp_path),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = build_server(
            tmp_path,
            port=0,
            coordinator_factory=lambda store: StudioCoordinator(
                store,
                qwen_factory=lambda run: ChairQwen(),
                comfy_factory=lambda run: FakeComfy(),
                worker_executor=FakeWorkerExecutor(),
                script_runner=FakeScriptRunner(),
                executor=executor,
            ),
        )
        thread = threading.Thread(target=running.server.serve_forever, daemon=True)
        thread.start()
        try:
            running.store.create("chair-page-v1", CHAIR_DESCRIPTION)
            running.coordinator.submit("chair-page-v1")
            _approve_gate(running.store, running.coordinator, "chair-page-v1", "D1")
            _approve_gate(running.store, running.coordinator, "chair-page-v1", "D8")
            run = _wait_for_stage_settled(running.store, "chair-page-v1", "D10")
            assert run.stage("D10").state == "awaiting_review", run.stage("D10").error

            assert any(item.relative_path.endswith(".glb") for item in run.stage("D2").evidence)
            status, d2_body, _ = _get(running, "/run/chair-page-v1/stage/D2")
            assert status == HTTPStatus.OK
            assert "data-glb-src=" in d2_body
            assert "Download original GLB" in d2_body

            status, body, _ = _get(running, "/run/chair-page-v1/stage/D10")
            assert status == HTTPStatus.OK
            assert "Open evidence" in body or "<img" in body
            for item in run.stage("D10").evidence:
                assert item.relative_path.rsplit("/", 1)[-1] in body
                # _get() decodes the body as UTF-8 text; the delivery board is
                # a PNG, so check the artifact route without decoding it.
                with urllib.request.urlopen(
                    f"{running.url}/artifact/chair-page-v1/{item.relative_path}", timeout=10
                ) as response:
                    assert response.status == HTTPStatus.OK
        finally:
            running.server.shutdown()
            running.server.server_close()
            thread.join(timeout=5)


def test_static_studio_js_is_served_same_origin_under_the_existing_csp(studio):
    """The polling script that replaced the meta-refresh reload must be a
    real same-origin file: the CSP has no 'unsafe-inline' on script-src, so
    an inline <script> block would be silently blocked by the browser with
    nothing for a server-side test to catch. This at least proves the file
    exists, is same-origin, and the CSP header is still intact."""
    status, body, headers = _get(studio, "/static/studio.js")
    assert status == HTTPStatus.OK
    assert headers["Content-Type"].startswith("text/javascript")
    assert "/api/run/" in body
    assert "location.reload" in body
    assert "X-Frame-Options" in headers
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_dependency_free_glb_viewer_is_served_same_origin(studio):
    status, body, headers = _get(studio, "/static/glb-viewer.js")
    assert status == HTTPStatus.OK
    assert headers["Content-Type"].startswith("text/javascript")
    assert "Only GLB 2.0 is supported" in body
    assert "data-glb-src" in body

    status, page, headers = _get(studio, "/")
    assert status == HTTPStatus.OK
    assert '<script src="/static/glb-viewer.js" defer>' in page
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_run_page_embeds_the_polling_script_only_while_busy(studio):
    """The script tag -- and the data attributes it reads -- should not be
    silently absent while genuinely running, and should not be pointlessly
    injected once a run is idle at a human gate."""
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run_id = _only_run_id(studio)
    run = wait_for(studio.store, run_id, "awaiting_review")
    status, body, _ = _get(studio, f"/run/{run_id}")
    assert status == HTTPStatus.OK
    assert f'data-run-id="{run_id}"' in body
    assert 'data-state="awaiting_review"' in body
    # idle at a human gate: nothing left to poll for
    assert "/static/studio.js" not in body
    assert 'id="active-stage-bar"' in body or "id=active-stage-bar" in body


def test_run_page_embeds_the_polling_script_while_genuinely_busy(tmp_path: Path) -> None:
    """The counterpart to test_run_page_embeds_the_polling_script_only_while_busy
    above: while a job is actually running, the page must carry the script
    tag and a data-state matching the in-progress state, or nothing will
    ever poll for the human to see the gate open. BlockingControlFakeComfy
    blocks D1's own automatic candidate render, which is enough to observe
    the run mid-flight without needing to reach awaiting_review first."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        comfy = BlockingControlFakeComfy()
        running = build_server(
            tmp_path,
            port=0,
            coordinator_factory=lambda store: StudioCoordinator(
                store,
                qwen_factory=lambda run: FakeQwen(),
                comfy_factory=lambda run: comfy,
                worker_executor=FakeWorkerExecutor(),
                executor=executor,
            ),
        )
        thread = threading.Thread(target=running.server.serve_forever, daemon=True)
        thread.start()
        run_id = None
        try:
            _post(running, "/runs", {"csrf": running.csrf, "description": DESCRIPTION, "profile": "simple"})
            run_id = _only_run_id(running)
            assert comfy.started.wait(timeout=3), "D1's concept render never reached ComfyUI"
            status, body, _ = _get(running, f"/run/{run_id}")
            assert status == HTTPStatus.OK
            assert "/static/studio.js" in body
            assert 'data-state="running"' in body
        finally:
            if run_id is not None:
                running.coordinator.stop(run_id)
                comfy.interrupted.wait(timeout=3)
            running.server.shutdown()
            running.server.server_close()
            thread.join(timeout=5)


def test_archiving_hides_a_run_from_the_dashboard_but_keeps_it_reachable(studio):
    _post(studio, "/runs", {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"})
    run_id = _only_run_id(studio)
    wait_for(studio.store, run_id, "awaiting_review")

    status, body = _post(studio, f"/run/{run_id}/archive", {"csrf": studio.csrf})
    assert status in {HTTPStatus.FOUND, HTTPStatus.SEE_OTHER}
    assert studio.store.load(run_id).archived is True

    status, body, _ = _get(studio, "/")
    assert status == HTTPStatus.OK
    assert run_id not in body
    assert "Show 1 archived run" in body

    status, body, _ = _get(studio, "/?archived=1")
    assert status == HTTPStatus.OK
    assert run_id in body
    assert "Hide archived runs" in body

    # the run's own page and API are untouched by archiving
    status, body, _ = _get(studio, f"/run/{run_id}")
    assert status == HTTPStatus.OK
    assert "Archived" in body

    status, body = _post(studio, f"/run/{run_id}/unarchive", {"csrf": studio.csrf})
    assert status in {HTTPStatus.FOUND, HTTPStatus.SEE_OTHER}
    assert studio.store.load(run_id).archived is False
    status, body, _ = _get(studio, "/")
    assert run_id in body


def test_a_dependency_looking_failure_gets_a_doctor_link(tmp_path: Path) -> None:
    """A failed stage's raw exception text used to render with no guidance
    beyond a bare Resume button, indistinguishable from a genuinely bad
    generation result. When the error looks like a missing local dependency
    it should now point at /doctor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = build_server(
            tmp_path,
            port=0,
            coordinator_factory=lambda store: StudioCoordinator(
                store,
                qwen_factory=lambda run: FakeQwen(),
                comfy_factory=lambda run: FakeComfy(),
                worker_executor=FakeWorkerExecutor(),
                executor=executor,
            ),
        )
        thread = threading.Thread(target=running.server.serve_forever, daemon=True)
        thread.start()
        try:
            _post(running, "/runs", {"csrf": running.csrf, "description": DESCRIPTION, "profile": "simple"})
            run_id = _only_run_id(running)
            wait_for(running.store, run_id, "awaiting_review")
            run = running.store.load(run_id)
            run.stage("D1").error = "RuntimeError: the configured Blender executable does not exist: C:/nope.exe"
            run.state = "failed"
            run.stage("D1").state = "failed"
            running.store.save(run)

            status, body, _ = _get(running, f"/run/{run_id}")
            assert status == HTTPStatus.OK
            assert "missing local dependency" in body
            assert "does not exist" in body

            status, body, _ = _get(running, f"/run/{run_id}/stage/D1")
            assert status == HTTPStatus.OK
            assert "missing local dependency" in body
        finally:
            running.server.shutdown()
            running.server.server_close()
            thread.join(timeout=5)


def test_an_ordinary_failure_does_not_get_a_spurious_doctor_link(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = build_server(
            tmp_path,
            port=0,
            coordinator_factory=lambda store: StudioCoordinator(
                store,
                qwen_factory=lambda run: FakeQwen(),
                comfy_factory=lambda run: FakeComfy(),
                worker_executor=FakeWorkerExecutor(),
                executor=executor,
            ),
        )
        thread = threading.Thread(target=running.server.serve_forever, daemon=True)
        thread.start()
        try:
            _post(running, "/runs", {"csrf": running.csrf, "description": DESCRIPTION, "profile": "simple"})
            run_id = _only_run_id(running)
            wait_for(running.store, run_id, "awaiting_review")
            run = running.store.load(run_id)
            run.stage("D1").error = "ValueError: candidate ranking did not include every rendered id"
            run.state = "failed"
            run.stage("D1").state = "failed"
            running.store.save(run)

            status, body, _ = _get(running, f"/run/{run_id}")
            assert status == HTTPStatus.OK
            # the nav header always links to /doctor; only the remediation
            # hint's own wording should be conditional on the error's shape
            assert "missing local dependency" not in body
        finally:
            running.server.shutdown()
            running.server.server_close()
            thread.join(timeout=5)


def test_missing_services_is_silent_for_injected_providers_and_reports_real_ones(
    tmp_path: Path,
) -> None:
    """A test or embedding caller supplies its own providers, so loopback
    readiness says nothing about that run -- but a coordinator driving real
    local services must report what is not answering, with a remedy."""
    from text2model_forge.studio_store import StudioStore

    store = StudioStore(tmp_path)
    dead = {"localdeploy_url": "http://127.0.0.1:9/v1", "comfy_url": "http://127.0.0.1:9"}

    faked = StudioCoordinator(store, qwen_factory=lambda run: None, comfy_factory=lambda run: None)
    assert faked.missing_services(dead) == []

    real = StudioCoordinator(store)
    missing = real.missing_services(dead)
    assert {item["name"] for item in missing} == {"Reviewer model", "ComfyUI"}
    assert all(item["remedy"] for item in missing)


def test_a_run_is_never_created_while_a_required_service_is_down(
    studio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported failure: Studio accepted a description while nothing was
    running, then marked D0 failed two seconds later and left a dead run
    behind. Nothing should be created at all."""
    monkeypatch.setattr(
        type(studio.coordinator),
        "missing_services",
        lambda self, settings: [
            {"name": "Reviewer model", "url": "http://127.0.0.1:11434/v1", "remedy": "ollama serve"}
        ],
    )
    status, body = _post(
        studio,
        "/runs",
        {"csrf": studio.csrf, "description": DESCRIPTION, "profile": "simple"},
    )
    assert status == HTTPStatus.OK
    assert "Start your local AI services first" in body
    assert "ollama serve" in body
    # The description survives so retrying costs one click, not retyping.
    assert html.escape(DESCRIPTION, quote=True) in body
    assert studio.store.list() == []
