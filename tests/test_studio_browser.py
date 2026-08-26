"""Browser-level smoke tests for responsive Studio flows.

The HTTP suite remains the fast contract test. These checks exercise the
rendered layout in Chromium when the development browser is installed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
import threading
import time

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from text2model_forge.studio_pipeline import StudioCoordinator
from text2model_forge.studio_web import build_server
from test_studio import DESCRIPTION, FakeComfy, FakeQwen, FakeWorkerExecutor


@pytest.fixture
def browser_studio(tmp_path: Path):
    executor = ThreadPoolExecutor(max_workers=1)
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
        running.coordinator.close()
        running.server.server_close()
        thread.join(timeout=5)
        executor.shutdown(wait=True, cancel_futures=True)


def _launch(runtime):
    try:
        return runtime.chromium.launch()
    except playwright.Error as exc:
        pytest.skip(f"Chromium is not installed for Playwright: {exc}")


def test_first_run_and_creation_flow_reflow_without_page_overflow(browser_studio) -> None:
    with playwright.sync_playwright() as runtime:
        browser = _launch(runtime)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(browser_studio.url, wait_until="networkidle")
        assert page.locator("h1").count() == 1
        assert page.get_by_role("button", name="Open offline walkthrough").is_visible()
        assert page.evaluate("document.documentElement.scrollWidth") == 1440
        for width in (390, 320):
            page.set_viewport_size({"width": width, "height": 844})
            page.reload(wait_until="networkidle")
            assert page.evaluate("document.documentElement.scrollWidth") == width
            assert page.get_by_role("button", name="Open offline walkthrough").is_visible()

        page.goto(browser_studio.url + "/new", wait_until="networkidle")
        assert page.get_by_label("Asset brief").is_visible()
        assert page.get_by_label("Intended use (optional)").is_visible()
        for width in (390, 320):
            page.set_viewport_size({"width": width, "height": 844})
            page.reload(wait_until="networkidle")
            assert page.evaluate("document.documentElement.scrollWidth") == width
            assert page.locator("form[data-setup-options] button[type=submit]").is_visible()
        browser.close()


def test_offline_walkthrough_has_an_obvious_completion_surface(browser_studio) -> None:
    with playwright.sync_playwright() as runtime:
        browser = _launch(runtime)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(browser_studio.url, wait_until="networkidle")
        with page.expect_response(
            lambda response: response.request.method == "POST"
            and response.url.endswith("/demo")
            and response.status == HTTPStatus.SEE_OTHER
        ):
            page.get_by_role("button", name="Open offline walkthrough").click()
        page.wait_for_load_state("networkidle")

        assert page.get_by_text("Offline tutorial.", exact=False).is_visible()
        assert page.get_by_role("link", name="Download final GLB").is_visible()
        for width in (390, 320):
            page.set_viewport_size({"width": width, "height": 844})
            page.reload(wait_until="networkidle")
            assert page.evaluate("document.documentElement.scrollWidth") == width
            assert page.get_by_role("link", name="Download final GLB").is_visible()
        browser.close()


def test_review_gate_remains_usable_at_desktop_and_phone_widths(browser_studio) -> None:
    with playwright.sync_playwright() as runtime:
        browser = _launch(runtime)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(browser_studio.url + "/new", wait_until="networkidle")
        page.get_by_label("Asset brief").fill(DESCRIPTION)
        with page.expect_navigation(wait_until="domcontentloaded"):
            page.locator("form[data-setup-options]").evaluate("form => form.submit()")

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            page.reload(wait_until="networkidle")
            if page.locator("#human-decision").count():
                break
            page.wait_for_timeout(100)
        assert page.locator("#human-decision").count() == 1
        assert page.get_by_role("button", name="Approve and continue").is_visible()
        assert page.get_by_role("button", name="Request revision").is_visible()
        unlabeled = page.evaluate(
            """[...document.querySelectorAll('input,select,textarea')]
            .filter(control => control.type !== 'hidden' && control.labels.length === 0).length"""
        )
        assert unlabeled == 0

        for width in (1440, 390, 320):
            page.set_viewport_size({"width": width, "height": 844})
            page.reload(wait_until="networkidle")
            assert page.evaluate("document.documentElement.scrollWidth") == width
            assert page.locator("#human-decision").evaluate("form => form.scrollWidth <= form.clientWidth")
            assert page.get_by_role("button", name="Approve and continue").is_visible()
        browser.close()
