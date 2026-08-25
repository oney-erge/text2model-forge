"""Local-first browser UI for Text2Model Forge Studio runs and human gates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import email
import html
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
from typing import Any, NamedTuple
from urllib.parse import parse_qs, quote, unquote, urlparse
import urllib.request
import uuid
import webbrowser

from .config import load_local_config, worker_binding
from .hardware import detect_hardware, recommend_stack
from .manifests import load_manifests, preflight
from .settings import profiles_dir, resolve_settings, studio_overrides
from .studio_models import utc_now
from .studio_application import StudioApplication
from .studio_pipeline import StudioCoordinator
from .studio_store import StudioConflictError, StudioStore
from .studio_ui_assets import MODERN_STYLE


STYLE = """
:root{color-scheme:dark;--bg:#0c1114;--panel:#182126;--panel2:#202c32;--line:#3a4a51;--text:#eee6d7;--muted:#a4afb0;--accent:#dc6837;--steel:#45728b;--ok:#62bd79;--bad:#e26d62;--wait:#d5a746}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% -20%,#28373c,#0c1114 46%);color:var(--text);font:15px/1.45 system-ui,sans-serif}header{position:sticky;top:0;z-index:4;padding:18px 28px;background:#0d1417ee;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:24px}header h1{font-size:20px;letter-spacing:.12em;margin:0}header a{color:#c8dce5;text-decoration:none}.wrap{max-width:1320px;margin:auto;padding:26px}h1,h2,h3{margin:0 0 12px}p{margin:8px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.card{background:linear-gradient(145deg,var(--panel),#141c20);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:0 12px 30px #0005}.hero{padding:28px}.muted{color:var(--muted)}.error{color:#ff9e95}.good{color:#a7e7b6}.warning{color:#f0cf79}.timeline{display:grid;grid-template-columns:repeat(auto-fit,minmax(88px,1fr));gap:6px;margin:18px 0;overflow:auto}.stage{min-height:88px;border:1px solid var(--line);border-radius:9px;padding:9px;background:#12191d}.stage strong{display:block}.stage small{display:block;color:var(--muted)}.stage.approved{border-color:#3d8050;background:#17281c}.stage.skipped{opacity:.56;border-style:dashed}.stage.running,.stage.queued{border-color:var(--steel)}.stage.awaiting_review{border-color:var(--wait);background:#292315}.stage.rejected,.stage.failed{border-color:var(--bad);background:#2a1918}.stage.blocked{border-color:#8a6e3a}.bar{height:5px;background:#0b1012;border-radius:4px;margin-top:10px;overflow:hidden}.bar span{display:block;height:100%;background:var(--accent)}label{display:block;color:var(--muted);margin:10px 0 5px}textarea,input{width:100%;padding:11px;border:1px solid var(--line);border-radius:7px;background:#0d1417;color:var(--text)}textarea{min-height:130px;resize:vertical}button,.button{display:inline-block;border:0;border-radius:7px;padding:10px 16px;margin:10px 8px 0 0;background:var(--steel);color:white;text-decoration:none;cursor:pointer}.primary{background:var(--accent)!important}.reject,.danger{background:#9d4039!important}.memory{background:#735c24!important}.secondary{background:#34454c!important}.evidence{position:relative}.evidence img{width:100%;max-height:620px;object-fit:contain;background:#0a0f11;border:1px solid var(--line);border-radius:8px}.choice{display:flex;gap:8px;align-items:center;margin:8px 0}.choice input{width:auto}.badge{display:inline-block;padding:3px 7px;border-radius:999px;background:#29373d;color:#cbd8dc;font-size:12px}.recommended{background:#49381c;color:#ffe09a}pre,code{background:#0b1012;border:1px solid #2b373c;border-radius:6px}pre{padding:12px;white-space:pre-wrap;overflow:auto}code{padding:2px 5px}.review ul{margin-top:5px}.events{max-height:330px;overflow:auto}.event{border-left:2px solid var(--line);padding:5px 0 5px 12px;margin:4px 0}.run-card h2 a{color:var(--text);text-decoration:none}.actions{display:flex;gap:8px;flex-wrap:wrap}.health{display:flex;gap:8px;flex-wrap:wrap}.health span{padding:5px 9px;border-radius:8px;background:#253239}.health .down{background:#4a2220}.console{border-color:#5a707a;background:linear-gradient(135deg,#1b2a31,#142025)}.console-head{display:flex;justify-content:space-between;gap:12px;align-items:start}.console-head h2{margin:0}.control-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:10px;margin-top:15px}.control-action{border:1px solid var(--line);border-radius:9px;padding:12px;background:#10191d}.control-action h3{font-size:14px;margin-bottom:5px}.control-action p{font-size:13px;min-height:39px}.status-line{margin:14px 0 0;padding:10px 12px;border-radius:8px;background:#10191d;border-left:3px solid var(--steel)}.status-line.stopping{border-left-color:var(--bad);background:#2b1b1a}.status-line.idle{border-left-color:var(--ok)}.prompt-meta{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.prompt-tools{display:flex;align-items:center;gap:2px;flex-wrap:wrap}.prompt-tools button{margin-top:10px}button:disabled{cursor:not-allowed;opacity:.5}
.timeline a.stage{color:inherit;text-decoration:none;display:block}.timeline a.stage:hover{border-color:var(--accent)}.stage.active{outline:2px solid var(--accent);outline-offset:1px}.badge.gate{background:#3b2f47;color:#dcc7f0;margin-top:4px}.badge.na{background:#2a3238;color:#8f9ba0}
.attempt{display:flex;align-items:baseline;gap:10px;margin:20px 0 8px}.attempt h3{margin:0}.evidence details{margin-top:8px}.evidence summary{cursor:pointer;color:var(--muted);font-size:13px}.metrics{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 0}
.decisions{list-style:none;padding:0;margin:6px 0 0}.decisions li{border-left:2px solid var(--line);padding:6px 0 6px 12px;margin:6px 0}.decisions .approve{border-left-color:var(--ok)}.decisions .reject,.decisions .rollback{border-left-color:var(--bad)}.decisions .edit,.decisions .retry{border-left-color:var(--steel)}.decisions .skip{border-left-color:var(--muted)}
.run-card .bar{margin:12px 0 6px}.needs{background:var(--wait)!important;color:#1b1405!important}.crumb{display:flex;gap:10px;align-items:center;margin-bottom:12px}.crumb a{color:#c8dce5}
.bar.overall{height:12px;margin:8px 0 4px}.bar span{background:linear-gradient(90deg,var(--accent),#f0a154);transition:width .35s ease}.progress-label{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:13px}select{width:100%;padding:11px;border:1px solid var(--line);border-radius:7px;background:#0d1417;color:var(--text)}.options{margin-top:18px;border:1px solid var(--line);border-radius:9px;padding:12px;background:#10191d}.options summary{cursor:pointer;font-weight:650}.option-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:4px 14px}.service-status{margin-top:10px;padding:9px 11px;border-radius:7px;background:#0d1417}.service-status.good{border-left:3px solid var(--ok)}.service-status.warning{border-left:3px solid var(--wait)}
.bar.gpu span{background:linear-gradient(90deg,var(--steel),#75bdd9)}
.glb-preview{display:block;width:100%;height:420px;touch-action:none;cursor:grab;background:radial-gradient(circle,#263238,#080c0e 70%);border:1px solid var(--line);border-radius:8px}.glb-preview:active{cursor:grabbing}.viewer-note{font-size:12px;color:var(--muted)}
header{border-bottom-color:var(--line)}header .logo{flex:none;border-radius:7px}header h1{letter-spacing:.08em;font-weight:650}header a{padding:6px 2px;border-bottom:2px solid transparent}header a:hover,header a:focus-visible{color:var(--text);border-bottom-color:var(--accent)}
.active-runs{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;max-width:46vw}
.active-run-chip{display:flex;align-items:center;gap:6px;padding:5px 10px;border-radius:999px;background:#10191d;border:1px solid var(--line);color:var(--text);text-decoration:none;font-size:12px;max-width:220px}
.active-run-chip:hover,.active-run-chip:focus-visible{border-color:var(--accent)}
.active-run-chip span.title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.active-run-chip small{color:var(--muted);white-space:nowrap}
.active-run-chip .dot{flex:none;width:7px;height:7px;border-radius:50%;background:var(--steel)}
.active-run-chip.awaiting_review .dot{background:var(--wait)}
.hint{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;background:#253239;color:var(--muted);font-size:11px;font-weight:700;font-style:normal;cursor:help;position:relative;margin-left:5px;vertical-align:middle}
.hint:hover,.hint:focus-visible{background:var(--steel);color:#fff;outline:none}
.hint .tip{visibility:hidden;opacity:0;position:absolute;bottom:calc(100% + 7px);left:0;width:230px;background:#0d1417;border:1px solid var(--line);border-radius:7px;padding:9px 11px;font-size:12px;font-weight:400;color:var(--text);line-height:1.45;z-index:5;box-shadow:0 8px 20px #0007;transition:opacity .12s ease}
.hint:hover .tip,.hint:focus-visible .tip{visibility:visible;opacity:1}
.checking{color:var(--muted);animation:pulse 1.6s ease-in-out infinite}
.bar.indeterminate{height:5px;overflow:hidden}
.bar.indeterminate span{width:38%;background:linear-gradient(90deg,transparent,var(--accent),transparent);animation:slide 1.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
@media(prefers-reduced-motion:reduce){.checking,.bar.indeterminate span{animation:none}.bar.indeterminate span{width:100%}}
@media(max-width:800px){.wrap{padding:15px}header{padding:14px}.hero{padding:18px}.active-runs{display:none}}
"""


# A small bolt: forging is a heat/spark process, and a bolt reads clearly
# at 16px in a browser tab, which an anvil or hammer silhouette does not.
# Shared between the header wordmark and the /favicon.ico route so both
# stay in sync with one edit.
_LOGO_SVG_BODY = (
    '<rect x="2" y="2" width="28" height="28" rx="7" fill="#182126" '
    'stroke="#45728b" stroke-width="2"/>'
    '<path d="M17 6 9 18h5l-1.5 8L22 14h-5l2-8z" fill="#dc6837"/>'
)
_FAVICON = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">{_LOGO_SVG_BODY}</svg>'
).encode("utf-8")


def _page(title: str, body: str) -> bytes:
    return (
        '<!doctype html><html lang="en"><head><meta charset=utf-8>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title>"
        '<link rel="icon" href="/favicon.ico" type="image/svg+xml">'
        f"<style>{MODERN_STYLE}</style></head>"
        '<body><a class="skip-link" href="#main-content">Skip to content</a>'
        '<header class="app-header"><div class="header-inner">'
        '<a class="brand" href="/" aria-label="Text2Model Forge Studio home">'
        f'<svg class=logo viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">{_LOGO_SVG_BODY}</svg>'
        '<span class="brand-text">Text2Model Forge</span><span class="brand-kicker">Studio</span></a>'
        '<nav class="desktop-nav" aria-label="Primary">'
        '<a href="/">Projects</a><a href="/new">Create</a><a href="/doctor">System</a>'
        '<details class="nav-popover"><summary>Developer</summary><div class="nav-popover-menu">'
        '<a href="/golden">Golden corpus</a><a href="/api/v1/projects">Projects API</a>'
        '</div></details></nav>'
        '<div id=active-runs class=active-runs aria-live=polite aria-label="Active projects"></div>'
        '<details class="mobile-nav"><summary aria-label="Open navigation">Menu</summary>'
        '<nav class="nav-popover-menu" aria-label="Mobile">'
        '<a href="/">Projects</a><a href="/new">Create</a><a href="/doctor">System</a>'
        '<a href="/golden">Golden corpus</a></nav></details>'
        '</div></header><main id="main-content" class="wrap">'
        f'{body}</main><script src="/static/glb-viewer.js" defer></script>'
        '<script src="/static/active-runs.js" defer></script></body></html>'
    ).encode("utf-8")


# Polls /api/run/<id> while a stage is running and updates the active stage's
# progress bar and status text in place. The CSP has no 'unsafe-inline' on
# script-src (default-src 'self' covers same-origin <script src>, not inline
# blocks), so this must be served as a real same-origin file, not embedded --
# see build_server()'s CSP header. A full <meta http-equiv=refresh> reload
# used to do this job, but it reset scroll position every five seconds on any
# long-running stage; this only reloads the page when run.state or the
# current stage actually changes, and otherwise just updates numbers in place.
STUDIO_JS = """
(function () {
  function percentage(value) { return Math.round(Math.max(0, Math.min(1, value)) * 100); }

  function overallProgress(run) {
    var applicable = run.stages.filter(function (stage) { return stage.applicable; });
    if (!applicable.length) return 0;
    return applicable.reduce(function (sum, stage) { return sum + stage.progress; }, 0) / applicable.length;
  }

  function initRunProgress() {
    var hero = document.querySelector('[data-run-id]');
    if (!hero) return;
    var runId = hero.getAttribute('data-run-id');
    var knownState = hero.getAttribute('data-state');
    var knownStage = hero.getAttribute('data-current-stage');

    function apply(run) {
      if (run.state !== knownState || run.current_stage !== knownStage) {
        window.location.reload();
        return;
      }
      var stage = null;
      for (var i = 0; i < run.stages.length; i++) {
        if (run.stages[i].stage_id === run.current_stage) { stage = run.stages[i]; break; }
      }
      if (!stage) return;
      var stagePercent = percentage(stage.progress);
      var bar = document.getElementById('active-stage-bar');
      if (bar) {
        bar.style.width = stagePercent + '%';
        bar.parentNode.setAttribute('aria-valuenow', String(stagePercent));
      }
      var overallPercent = percentage(overallProgress(run));
      var overall = document.getElementById('overall-run-bar');
      if (overall) {
        overall.style.width = overallPercent + '%';
        overall.parentNode.setAttribute('aria-valuenow', String(overallPercent));
      }
      var overallLabel = document.getElementById('overall-run-label');
      if (overallLabel) overallLabel.textContent = overallPercent + '% overall';
      var message = document.getElementById('stage-message');
      if (message) message.textContent = stage.message;
      var work = document.getElementById('stage-work');
      if (work) {
        var count = stage.progress_total ? ' · ' + stage.progress_current + '/' + stage.progress_total + ' ' + stage.progress_unit : '';
        work.textContent = stage.progress_phase + count;
      }
      var gpu = document.getElementById('gpu-run-bar');
      if (gpu && stage.gpu_total_gb) {
        var gpuUsed = stage.gpu_used_gb || 0;
        var gpuPercent = Math.round(gpuUsed / stage.gpu_total_gb * 100);
        gpu.style.width = gpuPercent + '%';
        gpu.parentNode.setAttribute('aria-valuenow', String(gpuPercent));
      }
      var gpuLabel = document.getElementById('gpu-run-label');
      if (gpuLabel && stage.gpu_total_gb) {
        gpuLabel.textContent = (stage.gpu_used_gb || 0).toFixed(2) + '/' + stage.gpu_total_gb.toFixed(2) + ' GiB';
      }
      var status = document.getElementById('status-line');
      if (status && stage.state === 'running') {
        status.textContent = 'Running ' + stage.stage_id + ' at ' + stagePercent + '%.';
      }
    }

    function poll() {
      fetch('/api/run/' + encodeURIComponent(runId), { cache: 'no-store' })
        .then(function (response) { return response.json(); })
        .then(function (run) { apply(run); window.setTimeout(poll, 4000); })
        .catch(function () { window.setTimeout(poll, 8000); });
    }

    window.setTimeout(poll, 4000);
  }

  function fillDatalist(id, values) {
    var list = document.getElementById(id);
    if (!list) return;
    while (list.firstChild) list.removeChild(list.firstChild);
    values.forEach(function (value) {
      var option = document.createElement('option');
      option.value = value;
      list.appendChild(option);
    });
  }

  // One condition, one sentence. This used to concatenate four independent
  // fragments -- two raw service details plus two "required files not all
  // detected" lines -- so a machine with nothing installed yet reported
  // what read as four separate errors for what is really one state: the
  // local AI services have not been started.
  function describeServices(data) {
    var down = [];
    if (!data.services.reviewer.ready) down.push('the reviewer model');
    if (!data.services.comfyui.ready) down.push('ComfyUI');
    if (down.length > 1) {
      return {
        ok: false,
        text: 'No local AI services are running yet, so a run cannot start. The System page '
          + 'lists exactly what to start.'
      };
    }
    if (down.length === 1) {
      return {
        ok: false,
        text: down[0].charAt(0).toUpperCase() + down[0].slice(1)
          + ' is not running, so a run cannot start. The System page has the command.'
      };
    }
    var installed = [];
    if (data.qwen_image_2512_ready) installed.push('Qwen Image 2512');
    if (data.z_image_turbo_ready) installed.push('Z-Image Turbo');
    var text = 'Local AI services are running: ' + data.checkpoints.length + ' checkpoint'
      + (data.checkpoints.length === 1 ? '' : 's') + ' and ' + data.review_models.length
      + ' reviewer model' + (data.review_models.length === 1 ? '' : 's') + ' available.';
    if (installed.length) text += ' Installed here: ' + installed.join(' and ') + '.';
    return { ok: true, text: text };
  }

  function initSetupOptions() {
    var form = document.querySelector('[data-setup-options]');
    if (!form) return;
    var profile = form.querySelector('[name="profile"]');
    var status = document.getElementById('setup-service-status');
    var submit = form.querySelector('button[type="submit"]');
    var submitLabel = submit ? submit.textContent : '';

    // A convenience, not the gate: POST /runs re-checks server-side and
    // refuses there. So if this probe cannot run at all, leave the button
    // enabled and let the authoritative check answer.
    function setSubmitEnabled(enabled) {
      if (!submit) return;
      submit.disabled = !enabled;
      submit.textContent = enabled ? submitLabel : 'Start your AI services first';
    }

    function load() {
      if (status) {
        status.className = 'service-status';
        status.textContent = 'Checking local AI services and installed models...';
      }
      fetch('/api/setup/options?profile=' + encodeURIComponent(profile.value), { cache: 'no-store' })
        .then(function (response) { return response.json(); })
        .then(function (data) {
          Object.keys(data.defaults).forEach(function (name) {
            var field = form.querySelector('[name="' + name + '"]');
            if (!field) return;
            if (field.tagName === 'SELECT' && field.options.length && field.options[0].value === '') {
              field.options[0].textContent = 'Profile default: ' + data.defaults[name];
            } else {
              field.setAttribute('placeholder', 'Profile default: ' + data.defaults[name]);
            }
          });
          fillDatalist('installed-checkpoints', data.checkpoints);
          fillDatalist('installed-review-models', data.review_models);
          var state = describeServices(data);
          if (status) {
            status.className = 'service-status ' + (state.ok ? 'good' : 'warning');
            status.textContent = state.text;
          }
          setSubmitEnabled(state.ok);
        })
        .catch(function () {
          if (status) {
            status.className = 'service-status warning';
            status.textContent = 'Could not inspect local AI services. You can still use profile defaults or type model names.';
          }
          setSubmitEnabled(true);
        });
    }

    // The advanced-options panel starts open or closed based on the
    // server-rendered default profile; switching profiles client-side
    // should keep that in sync (simple -> collapsed, anything else ->
    // open) without overriding a manual click the user makes afterward.
    function syncOptionsVisibility() {
      var details = form.querySelector('details.options');
      if (details) details.open = profile.value !== 'simple';
    }

    profile.addEventListener('change', function () {
      syncOptionsVisibility();
      load();
    });
    load();
  }

  initRunProgress();
  initSetupOptions();
})();
"""


# The System page's checks deliberately wait on services that may not be
# running -- one ComfyUI probe alone carries a 25 s timeout, and the whole
# report was measured at ~5.8 s even when every connection is refused
# instantly. Rendering that synchronously meant the browser sat on a blank
# request the entire time, so the nav link read as a dead button. The page
# now returns its shell immediately and fills itself in from /api/doctor,
# which runs exactly the same checks -- the wait is unchanged, but it is
# visible and the rest of Studio stays usable during it.
DOCTOR_JS = """
(function () {
  function initDoctor() {
    var shell = document.getElementById('doctor-shell');
    if (!shell) return;
    fetch('/api/doctor', { cache: 'no-store' })
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.text();
      })
      .then(function (fragment) { shell.outerHTML = fragment; })
      .catch(function (error) {
        var status = document.getElementById('doctor-status');
        if (!status) return;
        status.className = 'error';
        status.textContent = 'Could not run the system checks (' + error.message + '). Reload to try again.';
      });
  }
  initDoctor();
})();
"""


# Loaded on every page (see _page()), unlike STUDIO_JS which only loads on
# pages that actually need it (a busy run, the New-asset form). This has to
# be global: the whole point is that a run's progress stays one click away
# no matter which page you are looking at, not just its own.
ACTIVE_RUNS_JS = """
(function () {
  function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function initActiveRuns() {
    var container = document.getElementById('active-runs');
    if (!container) return;

    function render(runs) {
      container.innerHTML = runs.map(function (run) {
        var label = run.state === 'awaiting_review' ? 'needs you' : run.current_stage;
        return '<a class="active-run-chip ' + run.state + '" href="/run/' + encodeURIComponent(run.run_id) + '">' +
          '<span class=dot></span><span class=title>' + escapeHtml(run.title) + '</span>' +
          '<small>' + escapeHtml(label) + '</small></a>';
      }).join('');
    }

    function poll() {
      fetch('/api/active-runs', { cache: 'no-store' })
        .then(function (response) { return response.json(); })
        .then(function (runs) { render(runs); window.setTimeout(poll, 4000); })
        .catch(function () { window.setTimeout(poll, 8000); });
    }

    poll();
  }

  initActiveRuns();
})();
"""


# Dependency-free local GLB preview. It reads mesh POSITION/index accessors and
# draws a sampled, depth-sorted shaded view on Canvas 2D. This is intentionally
# a review aid rather than a replacement for Blender or the D10 runtime gate.
GLB_VIEWER_JS = r"""
(function () {
  function components(type) { return {SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16}[type] || 1; }
  function componentBytes(type) { return {5120:1,5121:1,5122:2,5123:2,5125:4,5126:4}[type]; }
  function readComponent(view, offset, type) {
    if (type===5120) return view.getInt8(offset); if (type===5121) return view.getUint8(offset);
    if (type===5122) return view.getInt16(offset,true); if (type===5123) return view.getUint16(offset,true);
    if (type===5125) return view.getUint32(offset,true); return view.getFloat32(offset,true);
  }
  function parseGlb(buffer) {
    var file=new DataView(buffer); if(file.getUint32(0,true)!==0x46546c67||file.getUint32(4,true)!==2) throw Error('Only GLB 2.0 is supported');
    var offset=12,json=null,binary=null;
    while(offset+8<=buffer.byteLength){var length=file.getUint32(offset,true),type=file.getUint32(offset+4,true);offset+=8;
      if(type===0x4e4f534a) json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,offset,length)));
      if(type===0x004e4942) binary=buffer.slice(offset,offset+length); offset+=length;}
    if(!json||!binary) throw Error('GLB has no JSON or binary chunk');
    function accessor(index){var a=json.accessors[index],b=json.bufferViews[a.bufferView],count=components(a.type),size=componentBytes(a.componentType),stride=b.byteStride||count*size;
      var data=new DataView(binary),start=(b.byteOffset||0)+(a.byteOffset||0),out=new Array(a.count*count);
      for(var i=0;i<a.count;i++) for(var c=0;c<count;c++) out[i*count+c]=readComponent(data,start+i*stride+c*size,a.componentType); return out;}
    var triangles=[];
    (json.meshes||[]).forEach(function(mesh){(mesh.primitives||[]).forEach(function(p){if(!p.attributes||p.attributes.POSITION===undefined||(p.mode!==undefined&&p.mode!==4))return;
      var xyz=accessor(p.attributes.POSITION),indices=p.indices===undefined?null:accessor(p.indices),count=indices?indices.length:xyz.length/3;
      var step=Math.max(3,Math.ceil(count/150000/3)*3);
      for(var i=0;i+2<count;i+=step){var ia=indices?indices[i]:i,ib=indices?indices[i+1]:i+1,ic=indices?indices[i+2]:i+2;
        triangles.push([[xyz[ia*3],xyz[ia*3+1],xyz[ia*3+2]],[xyz[ib*3],xyz[ib*3+1],xyz[ib*3+2]],[xyz[ic*3],xyz[ic*3+1],xyz[ic*3+2]]]);}});});
    if(!triangles.length) throw Error('No triangle POSITION data was found'); return triangles;
  }
  function start(canvas){var context=canvas.getContext('2d'),triangles,yaw=-0.6,pitch=-0.25,zoom=0.82,drag=null,status='Loading GLB preview…';
    function resize(){var ratio=Math.min(devicePixelRatio||1,2),rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,rect.width*ratio);canvas.height=Math.max(1,rect.height*ratio);draw();}
    function draw(){var w=canvas.width,h=canvas.height;context.clearRect(0,0,w,h);context.fillStyle='#9fb0b7';context.font=(14*(devicePixelRatio||1))+'px system-ui';
      if(!triangles){context.fillText(status,18,28);return;} var points=[],min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
      triangles.forEach(function(t){t.forEach(function(p){for(var k=0;k<3;k++){min[k]=Math.min(min[k],p[k]);max[k]=Math.max(max[k],p[k]);}});});
      var center=[(min[0]+max[0])/2,(min[1]+max[1])/2,(min[2]+max[2])/2],extent=Math.max(max[0]-min[0],max[1]-min[1],max[2]-min[2],1e-6),cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),scale=Math.min(w,h)*zoom/extent;
      function project(p){var x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],rx=cy*x+sy*z,rz=-sy*x+cy*z,ry=cx*y-sx*rz;rz=sx*y+cx*rz;return [w/2+rx*scale,h/2-ry*scale,rz];}
      triangles.forEach(function(t){var a=project(t[0]),b=project(t[1]),c=project(t[2]),cross=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);if(Math.abs(cross)<0.01)return;points.push({p:[a,b,c],z:(a[2]+b[2]+c[2])/3,light:Math.max(.16,Math.min(.9,.45+cross/(Math.abs(cross)+9000)*.4))});});
      points.sort(function(a,b){return a.z-b.z;});points.forEach(function(t){var shade=Math.round(95+t.light*105);context.beginPath();context.moveTo(t.p[0][0],t.p[0][1]);context.lineTo(t.p[1][0],t.p[1][1]);context.lineTo(t.p[2][0],t.p[2][1]);context.closePath();context.fillStyle='rgb('+Math.round(shade*.72)+','+Math.round(shade*.86)+','+shade+')';context.fill();context.strokeStyle='rgba(8,14,17,.18)';context.stroke();});}
    canvas.addEventListener('pointerdown',function(e){drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',function(e){if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];draw();});
    canvas.addEventListener('pointerup',function(){drag=null;});canvas.addEventListener('wheel',function(e){e.preventDefault();zoom=Math.max(.2,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
    fetch(canvas.dataset.glbSrc,{cache:'no-store'}).then(function(r){if(!r.ok)throw Error('HTTP '+r.status);return r.arrayBuffer();}).then(function(b){triangles=parseGlb(b);resize();}).catch(function(e){status='Preview unavailable: '+e.message;draw();});
    addEventListener('resize',resize);resize();
  }
  document.querySelectorAll('canvas[data-glb-src]').forEach(start);
})();
"""


def _probe(url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200, f"HTTP {response.status}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _json_get(url: str) -> tuple[Any | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return json.load(response), f"HTTP {response.status}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _form(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        handler.close_connection = True
        raise ValueError("the request had no readable length") from exc
    if length < 0 or length > 1_000_000:
        # The body is deliberately not read, so this connection can no longer
        # be trusted to start the next request at a message boundary. Closing
        # it keeps an oversized form from desynchronising a keep-alive
        # connection and corrupting whatever request follows.
        handler.close_connection = True
        raise ValueError("form is too large")
    parsed = parse_qs(handler.rfile.read(length).decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def _multipart_form(
    handler: BaseHTTPRequestHandler, *, max_bytes: int
) -> dict[str, str | tuple[str, bytes]]:
    """Parse one multipart/form-data POST body, the one request shape _form()
    cannot handle because it is not urlencoded.

    Built on the standard library's own MIME parser (email.message_from_bytes)
    rather than a hand-rolled boundary splitter or the deprecated cgi module:
    this is security-relevant request parsing, so correctness on malformed or
    adversarial input matters more than brevity. A plain field comes back as
    its decoded string value; a file field comes back as (filename, bytes).
    """
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.startswith("multipart/form-data"):
        raise ValueError("expected a multipart/form-data request")
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        handler.close_connection = True
        raise ValueError("the request had no readable length") from exc
    if length < 0 or length > max_bytes:
        # Same reasoning as _form()'s cap: do not read an oversized body,
        # and do not trust this connection for a further request afterward.
        handler.close_connection = True
        raise ValueError("upload is too large")
    body = handler.rfile.read(length)
    message = email.message_from_bytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    if not message.is_multipart():
        raise ValueError("malformed multipart request")
    result: dict[str, str | tuple[str, bytes]] = {}
    for part in message.get_payload():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        result[name] = (filename, payload) if filename is not None else payload.decode("utf-8", "replace")
    return result


def _slug() -> str:
    from datetime import datetime, timezone

    return "asset-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5]


def _artifact_url(run_id: str, relative_path: str) -> str:
    return f"/artifact/{quote(run_id)}/{quote(relative_path, safe='/')}"


def _stage_url(run_id: str, stage_id: str) -> str:
    return f"/run/{quote(run_id)}/stage/{quote(stage_id)}"


def _run_progress(run) -> float:
    """Normalized progress across stages that apply to this asset.

    A stage's own fractional progress counts while it is running, instead of
    making the dashboard look frozen until a whole stage settles. D0 is the
    only stage that can change applicability; excluded stages are therefore
    omitted once the compiled contract identifies them.
    """
    applicable = [stage for stage in run.stages if stage.applicable]
    if not applicable:
        return 0.0
    return sum(stage.progress for stage in applicable) / len(applicable)


def _progress_bar(
    value: float,
    *,
    label: str,
    bar_id: str | None = None,
    extra_class: str = "",
) -> str:
    percent = round(max(0.0, min(1.0, value)) * 100)
    identifier = f' id="{html.escape(bar_id)}"' if bar_id else ""
    return (
        f'<div class="bar overall {html.escape(extra_class)}" role="progressbar" aria-label="{html.escape(label)}" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
        f'<span{identifier} style="width:{percent}%"></span></div>'
    )


def _timeline(run, *, active: str | None = None, mark_progress_id: bool = False) -> str:
    """Every stage as a link to its own detail page.

    The timeline used to be inert, which meant an approved stage's evidence
    became unreachable in the browser the moment the run moved past it --
    the run page only ever renders the *current* stage.

    `mark_progress_id` gives the active tile's bar a stable id so STUDIO_JS
    can update its width without a full page reload. Only the run page (where
    `active` really is the run's live current_stage) passes it; the stage
    detail page's `active` is just whichever stage the human is looking at.
    """
    tiles = []
    for item in run.stages:
        classes = "stage " + item.state + (" active" if item.stage_id == active else "")
        if item.applicable:
            state_badge = f'<span class="badge">{html.escape(item.state.replace("_", " "))}</span>'
        else:
            state_badge = '<span class="badge na">not applicable</span>'
        gate = '<span class="badge gate">human gate</span>' if item.gate_required else ""
        bar_id = ' id="active-stage-bar"' if mark_progress_id and item.stage_id == active else ""
        percent = round(item.progress * 100)
        tiles.append(
            f'<a class="{classes}" href="{_stage_url(run.run_id, item.stage_id)}">'
            f"<strong>{item.stage_id}</strong><small>{html.escape(item.label)}</small>"
            f'{state_badge}{gate}<div class="bar" role="progressbar" aria-label="{html.escape(item.stage_id)} progress" '
            f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
            f'<span{bar_id} style="width:{percent}%"></span></div></a>'
        )
    columns = max(len(run.stages), 1)
    return (
        f'<div class="timeline" style="grid-template-columns:repeat({columns},minmax(88px,1fr))">'
        + "".join(tiles)
        + "</div>"
    )


def _spec(run) -> str:
    if run.spec is None:
        return '<p class="muted">Qwen has not compiled the description yet.</p>'
    equipment = "".join(
        "<li>"
        f"<strong>{html.escape(item.equipment_id)}</strong> — {html.escape(item.category)}, "
        f"{html.escape(item.side)}, <code>{html.escape(item.socket)}</code>, {html.escape(item.grip)}"
        "</li>"
        for item in run.spec.equipment
    )
    components = "".join(
        "<li>"
        f"<strong>{html.escape(item.component_id)}</strong> — {html.escape(item.role)}, "
        f"{html.escape(item.motion)} via {html.escape(item.connection)}"
        "</li>"
        for item in run.spec.components
    )
    anatomy = html.escape(run.spec.anatomy_family or "not applicable")
    height = f"{run.spec.height_m:.2f} m high" if run.spec.height_m is not None else "height not applicable"
    dimensions = " × ".join(f"{item:g}" for item in run.spec.dimensions_m) + " m"
    return (
        f"<p>{html.escape(run.spec.creative_direction)}</p>"
        f"<p><span class=badge>{html.escape(run.spec.asset_kind)}</span> "
        f"<span class=badge>{html.escape(run.spec.behavior)}</span> "
        f"<span class=badge>{anatomy}</span> <span class=badge>{height}</span> "
        f"<span class=badge>{dimensions}</span></p>"
        f"<h3>Component contract</h3><ul>{components or '<li>Single continuous asset</li>'}</ul>"
        f"<h3>Equipment contract</h3><ul>{equipment or '<li>None</li>'}</ul>"
        f"<h3>Animations / states</h3><p>{html.escape(', '.join(run.spec.animations) or 'Static')}</p>"
        f"<h3>Locked features</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in run.spec.locked_features)}</ul>"
    )


def _qwen_review(stage) -> str:
    if not stage.qwen_reviews:
        return '<p class="muted">No Qwen review yet.</p>'
    return _one_qwen_review(stage.qwen_reviews[-1])


def _one_qwen_review(review) -> str:
    return (
        '<div class="review">'
        f"<p>{html.escape(review.summary)}</p>"
        f"<p><span class=badge>confidence {review.confidence:.0%}</span> "
        f"<span class=badge>locked requirements {'pass' if review.hard_requirements_satisfied else 'fail'}</span> "
        f"<span class='badge recommended'>recommended {html.escape(review.recommended_evidence_id or 'none')}</span></p>"
        f"<h3>Strengths</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in review.strengths) or '<li>None recorded</li>'}</ul>"
        f"<h3>Issues</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in review.issues) or '<li>None recorded</li>'}</ul>"
        f"<h3>If rejected</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in review.recommended_changes) or '<li>Use the human comment.</li>'}</ul>"
        "</div>"
    )


# Metrics every evidence card carries for bookkeeping rather than for the
# reader; they are still in the full JSON, just not worth a badge each.
_ROUTINE_METRICS = frozenset({"iteration", "selectable", "role"})


def _metric_badges(metrics: dict) -> str:
    """The handful of values worth reading at a glance, above the full JSON."""
    interesting = [
        (key, value)
        for key, value in metrics.items()
        if key not in _ROUTINE_METRICS and value is not None and value != ""
    ]
    if not interesting:
        return ""
    shown = "".join(
        f"<span class=badge>{html.escape(str(key))} {html.escape(str(value))}</span>"
        for key, value in interesting[:6]
    )
    return f'<p class="metrics">{shown}</p>'


def _evidence_iteration(item, stage) -> int:
    """Which attempt produced this artefact.

    Evidence written without an explicit iteration belongs to the attempt in
    progress -- the same rule the flat renderer used when it decided whether
    an item was still selectable.
    """
    raw = item.metrics.get("iteration")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return stage.iteration
    return int(raw)


def _evidence(run, stage, *, allow_selection: bool = True) -> tuple[str, list[str]]:
    """One stage's evidence, grouped by attempt, newest attempt first.

    A gate that has been rejected and retried a few times accumulates every
    attempt's artefacts on the same stage. Rendered as one flat grid, the two
    candidates actually up for decision were indistinguishable from eight
    superseded ones, and only the radio buttons hinted at which was which.
    """
    recommended = stage.qwen_reviews[-1].recommended_evidence_id if stage.qwen_reviews else None
    groups: dict[int, list] = {}
    for item in stage.evidence:
        groups.setdefault(_evidence_iteration(item, stage), []).append(item)
    if not groups:
        return '<p class="muted">This stage has not produced any evidence yet.</p>', []

    choices: list[str] = []
    sections: list[str] = []
    for iteration in sorted(groups, reverse=True):
        current = iteration == stage.iteration
        cards = []
        for item in groups[iteration]:
            url = _artifact_url(run.run_id, item.relative_path)
            pick = (
                '<span class="badge recommended">Qwen pick</span>'
                if item.evidence_id == recommended
                else ""
            )
            details = (
                '<details class="inspection-drawer"><summary>Technical evidence details</summary>'
                f"<pre>{html.escape(json.dumps({**item.metrics, 'sha256': item.sha256}, indent=2))}</pre></details>"
            )
            if item.media_type.startswith("image/"):
                choice = ""
                if (
                    allow_selection
                    and current
                    and item.metrics.get("selectable") is True
                    and stage.state == "awaiting_review"
                ):
                    checked = " checked" if item.evidence_id == recommended else ""
                    choice = (
                        f'<label class=evidence-select><input form=human-decision type=radio name=selected_evidence_id '
                        f'value="{html.escape(item.evidence_id)}"{checked}>'
                        f"Use {html.escape(item.label)}</label>"
                    )
                    choices.append(choice)
                body = (
                    f'<a href="{url}" target=_blank><img loading=lazy src="{url}" '
                    f'alt="{html.escape(item.label)}"></a>'
                )
            elif item.media_type in {"model/gltf-binary", "model/gltf+json"} or item.relative_path.lower().endswith(".glb"):
                choice = ""
                body = (
                    f'<canvas class=glb-preview data-glb-src="{url}" aria-label="Interactive 3D preview of {html.escape(item.label)}"></canvas>'
                    '<p class=viewer-note>Drag to orbit; use the wheel to zoom. This local sampled preview is for inspection only.</p>'
                    f'<p><a class=button href="{url}" target=_blank>Download original GLB</a></p>'
                )
            else:
                choice = ""
                body = f'<p><a class=button href="{url}" target=_blank>Open evidence</a></p>'
            cards.append(
                '<section class="card evidence is-candidate">'
                f"<h3>{html.escape(item.label)} {pick}</h3>{body}"
                f"{_metric_badges(item.metrics)}{details}{choice}</section>"
            )
        heading = (
            f'<div class="attempt"><h3>Attempt {iteration}</h3>'
            + (
                '<span class="badge recommended">current attempt</span>'
                if current
                else '<span class=badge>superseded</span>'
            )
            + "</div>"
        )
        sections.append(heading + '<div class="evidence-grid">' + "".join(cards) + "</div>")
    return "".join(sections), choices


def _decision_history(stage) -> str:
    """The append-only human record for one stage.

    Gate decisions are the product's core claim, and until now they were
    only visible as raw JSON inside the last thirty run events -- which a
    long run pushes out entirely.
    """
    if not stage.human_decisions:
        return '<p class="muted">No human decision has been recorded at this gate yet.</p>'
    rows = []
    for item in stage.human_decisions:
        facts = [f"<span class=badge>{len(item.evidence_hashes)} artefacts hash-bound</span>"]
        if item.selected_evidence_id:
            facts.append(f"<span class=badge>selected {html.escape(item.selected_evidence_id)}</span>")
        if item.target_stage_id:
            facts.append(f"<span class=badge>target {html.escape(item.target_stage_id)}</span>")
        if item.overrides:
            facts.append(f"<span class=badge>overrides {html.escape(json.dumps(item.overrides))}</span>")
        if item.assisted_by_review_id:
            facts.append(
                f"<span class='badge recommended'>AI-assisted {html.escape(item.assisted_by_review_id)}</span>"
            )
        rows.append(
            f'<li class="{html.escape(item.decision)}"><strong>{html.escape(item.decision)}</strong> '
            f'<span class=muted>{html.escape(item.created_at.isoformat())}</span>'
            f"<p>{html.escape(item.comment) or '<span class=muted>No comment recorded.</span>'}</p>"
            f'<p class="metrics">{"".join(facts)}</p></li>'
        )
    return f'<ul class="decisions">{"".join(rows)}</ul>'


def _ai_recommendation_form(run, stage, csrf: str) -> str:
    """Render a review recommendation as an explicit human confirmation.

    The model never submits the gate. The review ID is posted only by this
    dedicated confirmation form, so an ordinary manual decision is not
    mislabeled as AI-assisted.
    """
    if stage.state != "awaiting_review" or not stage.qwen_reviews:
        return ""
    review = stage.qwen_reviews[-1]
    if review.stage_id != stage.stage_id or review.iteration != stage.iteration:
        return ""
    recommended_item = next(
        (
            item
            for item in stage.evidence
            if item.evidence_id == review.recommended_evidence_id
            and item.metrics.get("selectable") is True
        ),
        None,
    )
    decision = (
        "approve"
        if review.hard_requirements_satisfied and recommended_item is not None
        else "reject"
    )
    selected_input = (
        f'<input type=hidden name=selected_evidence_id value="{html.escape(recommended_item.evidence_id, quote=True)}">'
        if recommended_item is not None
        else ""
    )
    details = review.recommended_changes or review.issues
    comment = review.summary
    if details:
        comment += " Suggested changes: " + "; ".join(details)
    verdict = "Approve recommended candidate" if decision == "approve" else "Reject and iterate"
    return (
        '<section class="card review"><h2>AI review recommendation</h2>'
        f'<p><strong>{html.escape(verdict)}</strong> '
        f'<span class=badge>confidence {review.confidence:.0%}</span></p>'
        f'<p>{html.escape(review.summary)}</p>'
        '<p class=muted>This is advisory. Inspect the evidence, then confirm to create a human-owned, '
        'hash-bound gate decision.</p>'
        f'<form method=post action="/run/{html.escape(run.run_id, quote=True)}/decision">'
        f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
        f'<input type=hidden name=stage_id value="{html.escape(stage.stage_id, quote=True)}">'
        f'<input type=hidden name=decision value="{decision}">'
        f'<input type=hidden name=comment value="{html.escape(comment, quote=True)}">'
        f'<input type=hidden name=assisted_by_review_id value="{html.escape(review.review_id, quote=True)}">'
        f'{selected_input}<button class=primary type=submit>Confirm AI recommendation: {html.escape(verdict)}</button>'
        f'<small class=muted>Review {html.escape(review.review_id)}</small></form></section>'
    )


def _legacy_decision_form(run, stage, csrf: str, choices: list[str]) -> str:
    if stage.state != "awaiting_review":
        return ""
    stage_index = next(
        (i for i, item in enumerate(run.stages) if item.stage_id == stage.stage_id), 0
    )
    rollback_targets = [
        item
        for item in run.stages[:stage_index]
        # A stage the asset contract ruled out cannot be rolled back to: it
        # was never decided, and StudioStore.decide() would refuse it.
        if item.applicable and item.state in {"approved", "skipped", "rejected", "failed"}
    ]
    rollback_options = "".join(
        f'<option value="{item.stage_id}">{html.escape(item.stage_id)} -- {html.escape(item.label)}</option>'
        for item in rollback_targets
    )
    rollback_block = (
        f'<label>Roll back to</label><select name=target_stage_id><option value="">(not a rollback)</option>{rollback_options}</select>'
        if rollback_targets
        else '<input type=hidden name=target_stage_id value="">'
    )
    return _ai_recommendation_form(run, stage, csrf) + (
        '<section class="card"><h2>Your decision</h2><p>Approve a selected candidate, reject with a comment for '
        "Qwen's next attempt, retry the same stage fresh, edit with a concrete correction, skip this stage, or "
        'roll back to an earlier stage. Full history is preserved either way.</p>'
        f'<form method=post action="/run/{html.escape(run.run_id)}/decision">'
        f'<input type=hidden name=csrf value="{csrf}"><input type=hidden name=stage_id value="{stage.stage_id}">'
        + "".join(choices)
        + '<label>Comment</label><textarea name=comment placeholder="Required to reject or skip. An edit needs a '
        'comment, overrides, or both. Optional for approve, retry, and roll back."></textarea>'
        '<label>Overrides (JSON, optional -- used by retry and edit)</label>'
        '<textarea name=overrides placeholder=\'e.g. {"seed": 42}\'></textarea>'
        + rollback_block
        + '<div class=actions>'
        '<button class=primary name=decision value=approve type=submit>Approve and continue</button>'
        '<button class=reject name=decision value=reject type=submit>Reject and iterate</button>'
        '<button class=secondary name=decision value=retry type=submit>Retry this stage</button>'
        '<button class=secondary name=decision value=edit type=submit>Edit and retry</button>'
        '<button class=secondary name=decision value=skip type=submit>Skip this stage</button>'
        + (
            '<button class=danger name=decision value=rollback type=submit>Roll back</button>'
            if rollback_targets
            else ""
        )
        + '</div></form></section>'
    )


def _decision_form(run, stage, csrf: str, choices: list[str]) -> str:
    """Focused human gate with infrequent controls behind progressive disclosure."""
    if stage.state != "awaiting_review":
        return ""
    stage_index = next(
        (index for index, item in enumerate(run.stages) if item.stage_id == stage.stage_id),
        0,
    )
    rollback_targets = [
        item
        for item in run.stages[:stage_index]
        if item.applicable and item.state in {"approved", "skipped", "rejected", "failed"}
    ]
    rollback_options = "".join(
        f'<option value="{html.escape(item.stage_id, quote=True)}">'
        f'{html.escape(item.stage_id)}: {html.escape(item.label)}</option>'
        for item in rollback_targets
    )
    rollback_block = (
        '<label for=rollback-target>Roll back to</label>'
        f'<select id=rollback-target name=target_stage_id>'
        f'<option value="">Choose an earlier stage</option>{rollback_options}</select>'
        if rollback_targets
        else '<input type=hidden name=target_stage_id value="">'
    )
    rollback_button = (
        '<button class=danger name=decision value=rollback type=submit>Roll back</button>'
        if rollback_targets
        else ""
    )
    return _ai_recommendation_form(run, stage, csrf) + (
        '<section class=card><p class=eyebrow>Human gate</p><h2>Your decision</h2>'
        '<p class=muted>Inspect the evidence first. Your choice is stored with the exact evidence hashes.</p>'
        f'<form id=human-decision method=post action="/run/{html.escape(run.run_id, quote=True)}/decision">'
        f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
        f'<input type=hidden name=stage_id value="{html.escape(stage.stage_id, quote=True)}">'
        + '<label for=decision-comment>Review note</label>'
        '<textarea id=decision-comment name=comment '
        'placeholder="Say what should change. A note is required when requesting a revision or skipping."></textarea>'
        '<div class=decision-primary>'
        '<button class=primary name=decision value=approve type=submit>Approve and continue</button>'
        '<button class=reject name=decision value=reject type=submit>Request revision</button></div>'
        '<details class=decision-more><summary>More decision options</summary>'
        '<label for=decision-overrides>Technical overrides (JSON, optional)</label>'
        "<textarea id=decision-overrides name=overrides placeholder='For example: {&quot;seed&quot;: 42}'></textarea>"
        + rollback_block
        + '<div class=actions><button class=secondary name=decision value=retry type=submit>'
        'Retry unchanged</button><button class=secondary name=decision value=edit type=submit>'
        'Edit inputs and retry</button><button class=secondary name=decision value=skip type=submit>'
        'Mark not applicable</button>'
        + rollback_button
        + '</div></details></form></section>'
    )


def _manual_qwen_image_form(run, stage, csrf: str, *, busy: bool) -> str:
    """Offer a direct D1 candidate without bypassing the stored production contract."""
    if stage.stage_id != "D1":
        return ""
    if busy or stage.state != "awaiting_review":
        return (
            '<section class="card console"><h2>Direct Qwen Image 2512 render</h2>'
            '<p class=muted>A direct prompt is available when this concept gate is waiting for review. '
            'Use the controls below to monitor or stop a running job.</p></section>'
        )
    return (
        '<section class="card console"><div class=console-head><div><h2>Direct Qwen Image 2512 render</h2>'
        '<p class=muted>One controlled candidate, returned to this same human-review gate.</p></div>'
        '<span class="badge recommended">ready</span></div>'
        '<p>Try a specific visual direction without restarting the asset. Your words are stored as evidence; '
        'Qwen expands them against the typed asset contract, renders one new candidate, then critiques it before '
        'returning it here for your decision.</p>'
        '<div class=prompt-meta><span class=badge>Qwen Image 2512</span><span class=badge>1104 &times; 1472</span>'
        '<span class=badge>50 sampling steps</span><span class=badge>Qwen contract rewrite + critic</span></div>'
        f'<form method=post action="/run/{html.escape(run.run_id)}/qwen-image">'
        f'<input type=hidden name=csrf value="{csrf}">'
        '<label for=direct-prompt>Visual prompt</label><textarea id=direct-prompt name=prompt minlength=12 required '
        'placeholder="Example: hand-painted clockwork courier with a canvas satchel on its left side and a brass lantern in its right hand; neutral studio background, clear proportions."></textarea>'
        '<label for=direct-seed>Optional deterministic seed</label><input id=direct-seed type=number name=seed min=0 step=1 placeholder="Leave blank for a fresh random seed">'
        '<div class=prompt-tools><button class=primary type=submit>Render direct Qwen candidate</button>'
        '<button class=secondary type=reset>Clear prompt and seed</button></div></form></section>'
    )


def _upload_image_form(run, stage, csrf: str, *, busy: bool) -> str:
    """Offer an uploaded image as a D1 candidate, through the same gate as any other."""
    if stage.stage_id != "D1":
        return ""
    if busy or stage.state != "awaiting_review":
        return (
            '<section class="card console"><h2>Upload your own concept image</h2>'
            '<p class=muted>An upload is available when this concept gate is waiting for review.</p></section>'
        )
    return (
        '<section class="card console"><div class=console-head><div><h2>Upload your own concept image</h2>'
        '<p class=muted>Skip generation and supply the D1 candidate yourself.</p></div>'
        '<span class="badge recommended">ready</span></div>'
        '<p>Use art you already have instead of generating a new candidate. It goes through the same '
        f'isolation and quality gate every generated concept does{_hint("Studio isolates the subject by flood-filling a mostly green, edge-connected background inward from the border -- the same check every generated concept passes. An image on a plain white, textured, or busy background will likely fail it; the fastest fix is compositing your art onto a flat green backdrop before uploading.")} '
        '-- render or place your subject on a clean, mostly green background before uploading.</p>'
        f'<form method=post action="/run/{html.escape(run.run_id)}/upload-image" enctype=multipart/form-data>'
        f'<input type=hidden name=csrf value="{csrf}">'
        '<label for=concept-upload>Image file</label><input id=concept-upload type=file name=image accept="image/*" required>'
        '<div class=prompt-tools><button class=primary type=submit>Upload as a candidate</button></div>'
        '</form></section>'
    )


def _studio_controls(run, stage, coordinator: StudioCoordinator, csrf: str, *, busy: bool) -> str:
    stopping = coordinator.stopping(run.run_id)
    if stopping:
        status = "Stop requested. Waiting for the active worker to reach a safe stop point."
        status_class = "stopping"
    elif busy:
        status = f"Running {stage.stage_id} at {stage.progress:.0%}. This page updates automatically."
        status_class = ""
    else:
        status = "Idle. You can render, review, resume a blocked stage, or free ComfyUI model memory."
        status_class = "idle"
    stop_action = (
        f'<form method=post action="/run/{html.escape(run.run_id)}/stop"><input type=hidden name=csrf value="{csrf}">'
        '<button class=danger type=submit>Stop current Studio work</button></form>'
        if busy and not stopping
        else '<button class=danger type=button disabled>No active Studio job</button>'
    )
    memory_action = (
        f'<form method=post action="/run/{html.escape(run.run_id)}/gpu-free"><input type=hidden name=csrf value="{csrf}">'
        '<button class=memory type=submit>Unload ComfyUI models + free VRAM</button></form>'
        if not busy
        else '<button class=memory type=button disabled>Free VRAM after the job stops</button>'
    )
    archive_action = (
        f'<form method=post action="/run/{html.escape(run.run_id)}/unarchive">'
        f'<input type=hidden name=csrf value="{csrf}">'
        '<button class=secondary type=submit>Unarchive this run</button></form>'
        if run.archived
        else f'<form method=post action="/run/{html.escape(run.run_id)}/archive">'
        f'<input type=hidden name=csrf value="{csrf}">'
        '<button class=secondary type=submit>Archive this run</button></form>'
    )
    return (
        '<section class="card console"><div class=console-head><div><h2>Studio control console</h2>'
        '<p class=muted>Controls apply only to the local Text2Model Forge Studio job and local ComfyUI service.</p></div>'
        f'<span class=badge>{"stopping" if stopping else "running" if busy else "idle"}</span></div>'
        f'<div id=status-line class="status-line {status_class}">{html.escape(status)}</div>'
        '<div class=control-grid><section class=control-action><h3>Current work</h3>'
        '<p>Interrupts the tracked ComfyUI workflow and preserves all prior evidence. Resume is always explicit.</p>'
        f'{stop_action}</section><section class=control-action><h3>GPU / model memory</h3>'
        '<p>Unloads ComfyUI models and releases its execution cache. The next render will take longer while models reload.</p>'
        f'{memory_action}</section><section class=control-action><h3>Prompt workspace</h3>'
        '<p>The direct-render form includes a local Clear prompt and seed button. It never deletes stored evidence.</p>'
        '<p class=muted>ComfyUI is a shared local service: avoid unrelated Comfy jobs while stopping a Studio render.</p>'
        '</section><section class=control-action><h3>Dashboard visibility</h3>'
        '<p>Hides this run from the default dashboard list. Never deletes anything and is always reversible; '
        'the run stays reachable at its own URL either way.</p>'
        f'{archive_action}</section></div></section>'
    )


# Substrings that show up in an unmet-dependency error: a missing Blender
# executable, an absent config.local.toml worker binding, a ComfyUI model
# that was never installed. Heuristic, not exhaustive -- worst case a real
# bug's error just doesn't get the extra link, which is what happened for
# every failure before this existed.
#
# "urlopen error" is the one addition worth calling out: every network
# reachability check in this codebase goes through urllib, and URLError's
# own __str__ always renders as "<urlopen error ...>" regardless of the
# underlying OS message -- a Linux "Connection refused", a Windows
# "[WinError 10061] ... actively refused it", or a plain socket timeout.
# Matching that wrapper catches all of them at the source instead of
# enumerating each platform's wording one at a time.
_DEPENDENCY_ERROR_HINTS = (
    "not installed",
    "is required",
    "does not exist",
    "config.local.toml",
    "worker binding",
    "executable",
    "not reachable",
    "connection refused",
    "urlopen error",
    "actively refused",
)


def _error_block(stage) -> str:
    """A failed/blocked stage's raw error, plus what a human can actually do
    about it. Previously this was a bare <pre> of the exception text with no
    guidance beyond the Resume button already shown elsewhere on the page --
    fine for "the render itself failed," unhelpful for "ComfyUI was never
    started," which reads identically to a human until they already know to
    check /doctor."""
    if not stage.error:
        return ""
    hint = ""
    lowered = stage.error.lower()
    if any(needle in lowered for needle in _DEPENDENCY_ERROR_HINTS):
        hint = (
            '<p class=muted>This looks like a missing local dependency or configuration, not a bad '
            'result. Check <a href="/doctor">the system page</a> for what is and is not reachable, fix '
            "it, then use Resume below to retry this stage from its saved inputs -- no evidence is lost."
            "</p>"
        )
    return f'<pre class=error>{html.escape(stage.error)}</pre>{hint}'


def _duration_badge(stage) -> str:
    """started_at/finished_at are written on every stage transition -- eleven
    call sites in studio_pipeline.py -- but nothing ever read them back; no
    duration was reachable anywhere in the browser. Render what's available:
    a finished duration, an in-progress elapsed time, or neither."""
    if stage.started_at is None:
        return ""
    if stage.finished_at is not None:
        seconds = (stage.finished_at - stage.started_at).total_seconds()
        label = f"ran {seconds:.0f}s"
    elif stage.state == "running":
        seconds = (utc_now() - stage.started_at).total_seconds()
        label = f"running {seconds:.0f}s"
    else:
        return f'<span class=badge>started {html.escape(stage.started_at.isoformat(timespec="seconds"))}</span>'
    return f"<span class=badge>{html.escape(label)}</span>"


def _stage_page(store: StudioStore, run_id: str, stage_id: str) -> str:
    """One stage's full record, reachable at any time from the timeline.

    The run page shows only the current stage, so before this every earlier
    stage's evidence, Qwen reviews, and human decisions were unreachable from
    the browser as soon as the run advanced past them.
    """
    run = store.load(run_id)
    try:
        stage = run.stage(stage_id)
    except KeyError as exc:
        raise FileNotFoundError(f"unknown stage: {stage_id}") from exc
    evidence, _ = _evidence(run, stage, allow_selection=False)
    reviews = (
        "".join(
            f"<h3>Attempt {review.iteration} — {html.escape(review.review_id)}</h3>"
            + _one_qwen_review(review)
            for review in reversed(stage.qwen_reviews)
        )
        or '<p class="muted">Qwen did not review this stage.</p>'
    )
    gate = "Stops for your decision" if stage.gate_required else "Runs automatically"
    error = _error_block(stage)
    live = (
        f'<p><a class="button primary" href="/run/{quote(run_id)}">This stage is waiting for your '
        "decision — open the review gate</a></p>"
        if stage.state == "awaiting_review" and run.current_stage == stage.stage_id
        else ""
    )
    return (
        f'<div class=crumb><a href="/">Runs</a><span class=muted>/</span>'
        f'<a href="/run/{quote(run_id)}">{html.escape(run.title)}</a>'
        f'<span class=muted>/</span><span>{html.escape(stage.stage_id)}</span></div>'
        '<section class="card hero">'
        f"<h1>{html.escape(stage.stage_id)} — {html.escape(stage.label)}</h1>"
        f'<p><span class=badge>{html.escape(stage.state.replace("_", " "))}</span> '
        f"<span class=badge>{html.escape(gate)}</span> "
        f"<span class=badge>attempt {stage.iteration}</span> "
        f'<span class=badge>{"applies to this asset" if stage.applicable else "not applicable"}</span> '
        f"{_duration_badge(stage)}</p>"
        f'<div class=progress-label><span>Stage progress</span><span>{stage.progress:.0%}</span></div>'
        f'{_progress_bar(stage.progress, label=f"{stage.stage_id} progress")} '
        f"<p>{html.escape(stage.message)}</p>{error}{live}"
        f"{_timeline(run, active=stage.stage_id)}</section>"
        '<div class=grid style="margin-top:16px">'
        f'<section class=card><h2>Human decisions</h2>{_decision_history(stage)}</section>'
        '<section class=card><h2>Stage metrics</h2>'
        f'<pre>{html.escape(json.dumps(stage.metrics, indent=2, default=str))}</pre></section></div>'
        f'<section class="card" style="margin-top:16px"><h2>Qwen reviews</h2>{reviews}</section>'
        f"<h2 style='margin-top:22px'>Evidence</h2>{evidence}"
    )


def _legacy_run_page(store: StudioStore, coordinator: StudioCoordinator, run_id: str, csrf: str) -> str:
    run = store.load(run_id)
    stage = run.stage(run.current_stage)
    evidence, choices = _evidence(run, stage)
    events = store.read_events(run_id)[-30:]
    work_detail = html.escape(stage.progress_phase)
    if stage.progress_total:
        work_detail += (
            f" · {stage.progress_current}/{stage.progress_total} "
            f"{html.escape(stage.progress_unit)}"
        )
    gpu_fraction = (
        min(1.0, (stage.gpu_used_gb or 0.0) / stage.gpu_total_gb)
        if stage.gpu_total_gb
        else 0.0
    )
    gpu_label = (
        f"{stage.gpu_used_gb or 0.0:.2f}/{stage.gpu_total_gb:.2f} GiB"
        if stage.gpu_total_gb
        else "Waiting for telemetry"
    )
    gpu_progress = (
        '<div class=progress-label><span>Last observed GPU memory</span>'
        f'<span id=gpu-run-label>{gpu_label}</span></div>'
        f'{_progress_bar(gpu_fraction, label="Live GPU memory", bar_id="gpu-run-bar", extra_class="gpu")}'
    )
    event_html = "".join(
        f'<div class=event><strong>{html.escape(item["event_type"])}</strong> '
        f'<span class=muted>{html.escape(item["occurred_at"])}</span><br>'
        f'<small>{html.escape(json.dumps(item["payload"], default=str))}</small></div>'
        for item in reversed(events)
    )
    busy = coordinator.busy(run_id) or run.state == "running"
    actions = ""
    if run.state in {"failed", "blocked", "created"} and not coordinator.busy(run_id):
        actions = (
            f'<form method=post action="/run/{html.escape(run_id)}/resume"><input type=hidden name=csrf value="{csrf}">'
            '<button class=primary type=submit>Resume pipeline</button></form>'
        )
    error = _error_block(stage)
    overall_progress = _run_progress(run)
    body = (
        f'<section class="card hero" data-run-id="{html.escape(run.run_id)}" '
        f'data-state="{html.escape(run.state)}" data-current-stage="{html.escape(run.current_stage)}">'
        f"<h1>{html.escape(run.title)}</h1><p>{html.escape(run.description)}</p>"
        f'<p><span class=badge>{html.escape(run.run_id)}</span> <span class=badge>{html.escape(run.state)}</span> '
        f'<span class=badge>concept backend: {html.escape(run.concept_backend)}</span> '
        f'<span class=badge>device policy: {html.escape(run.device_policy)}</span> '
        f'<span class=badge>created {html.escape(run.created_at.isoformat(timespec="seconds"))}</span>'
        + (' <span class=badge>Archived</span>' if run.archived else "")
        + "</p>"
        f'<div class=progress-label><span>Whole pipeline</span><span id=overall-run-label>{overall_progress:.0%} overall</span></div>'
        f'{_progress_bar(overall_progress, label="Whole pipeline progress", bar_id="overall-run-bar")}'
        f'<div class=progress-label><span>Active work</span><span id=stage-work>{work_detail}</span></div>'
        f'{gpu_progress}'
        f"{_timeline(run, active=stage.stage_id, mark_progress_id=True)}"
        f'<h2><a href="{_stage_url(run.run_id, stage.stage_id)}">{stage.stage_id} — '
        f"{html.escape(stage.label)}</a></h2>"
        + (f'<p>{_duration_badge(stage)}</p>' if stage.started_at else "")
        + f'<p id=stage-message>{html.escape(stage.message)}</p>{error}{actions}</section>'
        '<div class=grid style="margin-top:16px"><section class=card><h2>Qwen production contract</h2>'
        f"{_spec(run)}</section><section class=card><h2>Qwen gate review</h2>{_qwen_review(stage)}"
        f'<p class=muted><a href="{_stage_url(run.run_id, stage.stage_id)}">'
        f"See every review and decision recorded at this stage</a></p></section></div>"
        f"{_studio_controls(run, stage, coordinator, csrf, busy=busy)}"
        f"<h2 style='margin-top:22px'>Evidence</h2>{evidence}"
        f"{_manual_qwen_image_form(run, stage, csrf, busy=busy)}"
        f"{_upload_image_form(run, stage, csrf, busy=busy)}"
        f"{_decision_form(run, stage, csrf, choices)}"
        f'<section class=card style="margin-top:16px"><h2>Run history</h2><div class=events>{event_html}</div></section>'
        + ('<script src="/static/studio.js" defer></script>' if busy else "")
    )
    return body


_USER_PHASES = (
    ("Brief", ("D0",)),
    ("Concept", ("D1",)),
    ("Geometry", ("D2", "D3")),
    ("Structure", ("D4", "D5", "D6", "D7")),
    ("Surface", ("D8", "D9")),
    ("Validate", ("D10",)),
)


def _phase_rail(run, active_stage_id: str) -> str:
    rows = []
    for number, (label, stage_ids) in enumerate(_USER_PHASES, start=1):
        phase_stages = [item for item in run.stages if item.stage_id in stage_ids]
        applicable = [item for item in phase_stages if item.applicable]
        is_active = active_stage_id in stage_ids
        if any(item.state == "awaiting_review" for item in applicable):
            state = "awaiting_review"
            state_text = "Needs review"
        elif any(item.state in {"failed", "blocked", "rejected"} for item in applicable):
            state = "failed"
            state_text = "Stopped"
        elif applicable and all(item.state in {"approved", "skipped"} for item in applicable):
            state = "approved"
            state_text = "Complete"
        elif any(item.state in {"running", "queued"} for item in applicable):
            state = "running"
            state_text = "In progress"
        elif not applicable:
            state = "skipped"
            state_text = "Not needed"
        else:
            state = "pending"
            state_text = "Upcoming"
        target = next(
            (item for item in reversed(phase_stages) if item.applicable),
            phase_stages[0],
        )
        href = (
            f"/run/{quote(run.run_id)}"
            if is_active
            else _stage_url(run.run_id, target.stage_id)
        )
        classes = f"phase-link {state}" + (" active" if is_active else "")
        rows.append(
            f'<a class="{classes}" href="{href}"><span class=phase-dot>{number}</span>'
            f'<span><strong>{html.escape(label)}</strong><small>{html.escape(state_text)}</small></span></a>'
        )
    return (
        '<section class=card><p class=eyebrow>Workflow</p>'
        f'<nav class=phase-list aria-label="Project phases">{"".join(rows)}</nav></section>'
    )


def _completion_panel(run) -> str:
    final_stage = run.stage("D10")
    models = [
        item
        for item in final_stage.evidence
        if item.media_type in {"model/gltf-binary", "model/gltf+json"}
        or item.relative_path.lower().endswith(".glb")
    ]
    final_model = models[-1] if models else None
    links = "".join(
        f'<a class="button secondary" href="{_artifact_url(run.run_id, item.relative_path)}" target=_blank>'
        f'Open {html.escape(item.label)}</a>'
        for item in final_stage.evidence
        if item is not final_model
    )
    if final_model is None:
        preview = '<p class=muted>The completion record does not contain a final GLB.</p>'
        download = ""
    else:
        url = _artifact_url(run.run_id, final_model.relative_path)
        preview = (
            f'<canvas class=glb-preview data-glb-src="{url}" '
            f'aria-label="Interactive preview of {html.escape(final_model.label, quote=True)}"></canvas>'
            '<p class=viewer-note>Drag to orbit and use the wheel to zoom.</p>'
        )
        download = f'<a class="button primary" href="{url}" download>Download final GLB</a>'
    return (
        '<section class="card completion"><p class=eyebrow>Build complete</p>'
        '<h2>Your validated asset is ready</h2>'
        '<p>The final package remains connected to the evidence and decisions that produced it.</p>'
        f'{preview}<div class=hero-actions>{download}{links}</div></section>'
    )


def _run_page(store: StudioStore, coordinator: StudioCoordinator, run_id: str, csrf: str) -> str:
    run = store.load(run_id)
    stage = run.stage(run.current_stage)
    evidence, choices = _evidence(run, stage)
    events = store.read_events(run_id)[-30:]
    busy = coordinator.busy(run_id) or run.state == "running"
    work_detail = html.escape(stage.progress_phase)
    if stage.progress_total:
        work_detail += (
            f" · {stage.progress_current}/{stage.progress_total} "
            f"{html.escape(stage.progress_unit)}"
        )
    gpu_fraction = (
        min(1.0, (stage.gpu_used_gb or 0.0) / stage.gpu_total_gb)
        if stage.gpu_total_gb
        else 0.0
    )
    gpu_label = (
        f"{stage.gpu_used_gb or 0.0:.2f}/{stage.gpu_total_gb:.2f} GiB"
        if stage.gpu_total_gb
        else "Waiting for telemetry"
    )
    overall_progress = _run_progress(run)
    error = _error_block(stage)
    resume = ""
    if run.state in {"failed", "blocked", "created"} and not coordinator.busy(run_id):
        resume = (
            f'<form method=post action="/run/{html.escape(run_id, quote=True)}/resume">'
            f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
            '<button class=primary type=submit>Resume build</button></form>'
        )
    tutorial_notice = (
        '<div class="notice tutorial"><strong>Offline tutorial.</strong> This completed example uses '
        'deterministic synthetic evidence. It teaches the interface and is not qualification evidence.</div>'
        if run.run_mode == "tutorial"
        else ""
    )
    event_html = "".join(
        f'<div class=event><strong>{html.escape(item["event_type"])}</strong> '
        f'<span class=muted>{html.escape(item["occurred_at"])}</span><br>'
        f'<small>{html.escape(json.dumps(item["payload"], default=str))}</small></div>'
        for item in reversed(events)
    )
    completion = _completion_panel(run) if run.state == "completed" else ""
    stage_status = (
        "Waiting for your decision"
        if stage.state == "awaiting_review"
        else stage.state.replace("_", " ").title()
    )
    active_card = (
        '<section class=card>'
        f'<div class=section-head><div><p class=eyebrow>Current stage</p>'
        f'<h2><a href="{_stage_url(run.run_id, stage.stage_id)}">'
        f'{html.escape(stage.stage_id)}: {html.escape(stage.label)}</a></h2></div>'
        f'<span class="badge{" needs" if stage.state == "awaiting_review" else ""}">'
        f'{html.escape(stage_status)}</span></div>'
        f'<p id=stage-message>{html.escape(stage.message)}</p>{error}{resume}'
        f'<div class=progress-label><span>Active work</span><span id=stage-work>{work_detail}</span></div>'
        f'{_progress_bar(stage.progress, label=f"{stage.stage_id} progress", bar_id="active-stage-bar")}'
        '<div class=progress-label><span>Last observed GPU memory</span>'
        f'<span id=gpu-run-label>{gpu_label}</span></div>'
        f'{_progress_bar(gpu_fraction, label="Live GPU memory", bar_id="gpu-run-bar", extra_class="gpu")}'
        '</section>'
    )
    alternative_concepts = ""
    if stage.stage_id == "D1":
        alternative_concepts = (
            '<details class="card advanced-drawer"><summary>Add another concept candidate</summary>'
            f'{_manual_qwen_image_form(run, stage, csrf, busy=busy)}'
            f'{_upload_image_form(run, stage, csrf, busy=busy)}</details>'
        )
    decision = _decision_form(run, stage, csrf, choices)
    review = (
        f'{decision}<section class=card><h2>Review notes</h2>{_qwen_review(stage)}</section>'
        '<section class=card><details><summary><strong>Production contract</strong></summary>'
        f'{_spec(run)}</details></section>'
    )
    evidence_block = (
        '<div class=evidence-heading><p class=eyebrow>Evidence</p><h2>Inspect the result</h2></div>'
        f'{evidence}{alternative_concepts}'
    )
    if run.state == "completed":
        active_card = (
            '<section class=card><div class=section-head><div><p class=eyebrow>Validation</p>'
            f'<h2>{html.escape(stage.stage_id)}: {html.escape(stage.label)}</h2></div>'
            '<span class=badge>Complete</span></div>'
            f'<p>{html.escape(stage.message)}</p>'
            f'<p><a href="{_stage_url(run.run_id, stage.stage_id)}">Open the complete D10 record</a></p>'
            '</section>'
        )
        evidence_block = (
            '<section class=card><p class=eyebrow>Provenance</p><h2>The evidence record is preserved</h2>'
            '<p>Open the technical stage record to inspect validation metrics, content hashes, and '
            'the append-only project history. The final download above is the delivery surface.</p>'
            f'<a class="button secondary" href="{_stage_url(run.run_id, stage.stage_id)}">'
            'Inspect validation evidence</a></section>'
        )
        review = (
            '<section class=card><p class=eyebrow>Project record</p><h2>Completed</h2>'
            f'<p>{len(stage.evidence)} final evidence item{"s" if len(stage.evidence) != 1 else ""} recorded.</p>'
            '</section><section class=card><details><summary><strong>Production contract</strong></summary>'
            f'{_spec(run)}</details></section>'
        )
    technical = (
        '<details class="card advanced-drawer"><summary>Technical stages and project controls</summary>'
        '<p class=muted>The D0 to D10 record remains available for audits and debugging.</p>'
        f'{_timeline(run, active=stage.stage_id, mark_progress_id=False)}'
        f'{_studio_controls(run, stage, coordinator, csrf, busy=busy)}'
        f'<h2>Run history</h2><div class=events>{event_html}</div></details>'
    )
    return (
        '<div class=crumb><a href=/>&larr; Projects</a></div>'
        f'<section class="card hero run-header" data-run-id="{html.escape(run.run_id, quote=True)}" '
        f'data-state="{html.escape(run.state, quote=True)}" '
        f'data-current-stage="{html.escape(run.current_stage, quote=True)}">'
        '<div class=run-summary><p class=eyebrow>3D project</p>'
        f'<h1>{html.escape(run.title)}</h1><p class=lede>{html.escape(run.description)}</p>'
        f'<p><span class=badge>{html.escape(run.state)}</span>'
        + (' <span class=badge>Offline tutorial</span>' if run.run_mode == "tutorial" else "")
        + (' <span class=badge>Archived</span>' if run.archived else "")
        + f' <span class=badge>{html.escape(run.run_id)}</span>'
        + f' <span class=badge>created {html.escape(run.created_at.isoformat(timespec="seconds"))}</span>'
        + (f' {_duration_badge(stage)}' if stage.started_at else "")
        + '</p>'
        '<div class=progress-label><span>Whole pipeline</span>'
        f'<span id=overall-run-label>{overall_progress:.0%} overall</span></div>'
        f'{_progress_bar(overall_progress, label="Whole pipeline progress", bar_id="overall-run-bar")}'
        f'{tutorial_notice}</div></section>'
        f'{completion}<div class=run-workspace><aside class=stage-rail>{_phase_rail(run, stage.stage_id)}</aside>'
        f'<div class=workspace-main>{active_card}{evidence_block}</div>'
        f'<aside class=review-panel>{review}</aside></div>'
        f'{technical}'
        + ('<script src="/static/studio.js" defer></script>' if busy else "")
    )


def _run_card(run) -> str:
    settled = sum(1 for item in run.stages if item.state in {"approved", "skipped"})
    total = max(len(run.stages), 1)
    overall_progress = _run_progress(run)
    waiting = run.state == "awaiting_review"
    attention = (
        '<span class="badge needs">Needs your decision</span>'
        if waiting
        else (
            '<span class="badge" style="background:#4a2220">Stopped</span>'
            if run.state in {"failed", "blocked"}
            else f"<span class=badge>{html.escape(run.state)}</span>"
        )
    )
    description = run.description if len(run.description) <= 240 else run.description[:237] + "..."
    stage = run.stage(run.current_stage) if any(
        item.stage_id == run.current_stage for item in run.stages
    ) else run.stages[0]
    archived_badge = '<span class=badge>Archived</span> ' if run.archived else ""
    tutorial_badge = (
        '<span class="badge">Offline tutorial</span> ' if run.run_mode == "tutorial" else ""
    )
    preview = next(
        (
            item
            for stage_item in reversed(run.stages)
            for item in reversed(stage_item.evidence)
            if item.media_type.startswith("image/")
        ),
        None,
    )
    thumbnail = (
        f'<a href="/run/{quote(run.run_id)}"><img class="project-thumb" loading="lazy" '
        f'src="{_artifact_url(run.run_id, preview.relative_path)}" '
        f'alt="Preview for {html.escape(run.title, quote=True)}"></a>'
        if preview is not None
        else '<div class="project-thumb project-thumb-empty" aria-hidden="true">◇</div>'
    )
    return (
        '<section class="card run-card">'
        f'{thumbnail}'
        f'<h2><a href="/run/{quote(run.run_id)}">{html.escape(run.title)}</a></h2>'
        f'<p>{archived_badge}{tutorial_badge}{attention} <span class=badge>{html.escape(stage.stage_id)} '
        f"{html.escape(stage.label)}</span> <span class=badge>{settled}/{total} stages settled</span></p>"
        f'<div class=progress-label><span>Pipeline progress</span><span>{overall_progress:.0%}</span></div>'
        f'{_progress_bar(overall_progress, label=f"{run.title} pipeline progress")}'
        f"<p>{html.escape(description)}</p>"
        f'<p class=muted>{html.escape(run.run_id)} · profile {html.escape(run.profile)} · '
        f"updated {html.escape(run.updated_at.isoformat(timespec='seconds'))}</p>"
        f'<div class="card-action"><a class="button{" primary" if waiting else ""}" href="/run/{quote(run.run_id)}">'
        f'{"Review now" if waiting else "Open run"}</a></div></section>'
    )


def _dashboard(store: StudioStore, csrf: str, *, show_archived: bool = False) -> str:
    # Runs arrive newest-first; float the ones blocked on a human above them,
    # because in a human-gated compiler an idle gate is the only thing that
    # actually stops the machine. Archived runs are hidden by default -- see
    # StudioRun.archived -- but never dropped from StudioStore.list() itself,
    # only from what this page chooses to show.
    all_runs = store.list()
    archived_count = sum(1 for run in all_runs if run.archived)
    runs = all_runs if show_archived else [run for run in all_runs if not run.archived]
    runs = sorted(runs, key=lambda item: item.state != "awaiting_review")
    waiting = sum(1 for run in runs if run.state == "awaiting_review")
    toggle = ""
    if archived_count:
        toggle = (
            '<p class=muted><a href="/?archived=1">Show ' + str(archived_count) + " archived run"
            + ("" if archived_count == 1 else "s") + "</a></p>"
            if not show_archived
            else '<p class=muted><a href="/">Hide archived runs</a></p>'
        )
    if not runs:
        return (
            '<section class="card hero empty-state"><div><p class=eyebrow>Local 3D production workspace</p>'
            '<span class=visually-hidden>Asset production runs</span>'
            '<h1>Turn a written brief into reviewable 3D evidence</h1>'
            '<p class=lede>Start with a guided offline project, configure your local generation stack, '
            'or inspect this computer before creating anything.</p>'
            f'{toggle}'
            '<div class=choice-grid>'
            '<article class=choice-card><div class=choice-icon aria-hidden=true>▶</div>'
            '<h2>Learn the workflow</h2><p>Open a completed cargo-crate project with real local files, '
            'review decisions, and a downloadable GLB. No model service is required.</p>'
            '<form method=post action=/demo><input type=hidden name=csrf '
            f'value="{html.escape(csrf, quote=True)}"><button class="primary block" type=submit>'
            'Open offline walkthrough</button></form></article>'
            '<article class=choice-card><div class=choice-icon aria-hidden=true>＋</div>'
            '<h2>Create an asset</h2><p>Describe the result you need. Studio preserves every candidate, '
            'review, and human gate as the project moves forward.</p>'
            '<a class="button block" href=/new>Configure and create</a></article>'
            '<article class=choice-card><div class=choice-icon aria-hidden=true>✓</div>'
            '<h2>Check your system</h2><p>See which local services, models, and tools are ready before '
            'you start a production project.</p>'
            '<a class="button secondary block" href=/doctor>Check this computer</a></article>'
            '</div></div></section>'
        )

    attention_runs = [run for run in runs if run.state == "awaiting_review"]
    recent_runs = [run for run in runs if run.state != "awaiting_review"]
    attention = (
        '<div class=section-head><div><p class=eyebrow>Needs review</p><h2>Your decision is next</h2></div>'
        f'<p>{len(attention_runs)} waiting for your decision</p></div>'
        '<div class=project-grid>' + "".join(_run_card(run) for run in attention_runs) + "</div>"
        if attention_runs
        else ""
    )
    shown_recent = recent_runs or ([] if attention_runs else runs)
    recent = (
        '<div class=section-head><div><p class=eyebrow>Projects</p><h2>Recent work</h2></div>'
        f'<p>{len(runs)} active and visible</p></div><div class=project-grid>'
        + "".join(_run_card(run) for run in shown_recent)
        + "</div>"
        if shown_recent
        else ""
    )
    return (
        '<section class=dashboard-shell><div class=dashboard-main>'
        '<section class="card hero"><p class=eyebrow>Project dashboard</p><h1>Your 3D projects</h1>'
        '<p class=lede>Continue a review, inspect completed evidence, or start from a new brief.</p>'
        f'{toggle}</section>{attention}{recent}</div>'
        '<aside class=dashboard-aside aria-label="Project actions">'
        '<section class=card><h2>Create</h2><p class=muted>Start a project from a written brief and '
        'choose the local tools that will produce it.</p><a class="button primary block" href=/new>'
        'New asset</a></section>'
        '<section class=card><h2>Workspace</h2><p><a href=/doctor>System readiness</a></p>'
        '<p><a href=/golden>Developer golden corpus</a></p></section>'
        '</aside></section>'
    )


def _active_runs_payload(store: StudioStore) -> list[dict[str, Any]]:
    """Runs genuinely in flight or waiting on a human, for the header strip.

    Deliberately narrower than the dashboard: "created" hasn't produced
    anything yet, "blocked"/"failed"/"completed" are stopped states the
    dashboard already surfaces. Archived runs are excluded for the same
    reason the dashboard hides them by default -- the user said they were
    done with it.
    """
    active = [
        run
        for run in store.list()
        if not run.archived and run.state in {"running", "awaiting_review"}
    ]
    active.sort(key=lambda run: run.state != "awaiting_review")
    return [
        {
            "run_id": run.run_id,
            "title": run.title,
            "state": run.state,
            "current_stage": run.current_stage,
        }
        for run in active
    ]


def _available_profiles() -> list[str]:
    directory = profiles_dir()
    if not directory.is_dir():
        return ["simple"]
    names = sorted(p.stem for p in directory.glob("*.toml") if p.stem != "base")
    return names or ["simple"]


def _service_unavailable_detail(remedy: str) -> str:
    """A human status for the New-asset service widget's offline case.

    _json_get's raw failure text is "<ExceptionType>: <message>", e.g.
    "URLError: <urlopen error timed out>" or, on Windows, "URLError:
    <urlopen error [WinError 10061] ... actively refused it>". That is
    accurate and meaningless to someone who just typed a prompt -- it
    names neither what is missing nor what to do. This names both.
    """
    return f"not running or not reachable -- {remedy}. See /doctor for the full check."


def _setup_options(profile: str) -> dict[str, Any]:
    """Profile defaults plus live model choices for the new-run form.

    Service discovery is advisory: an offline ComfyUI or reviewer returns an
    empty list and a visible status, while the form remains usable with typed
    model ids. This keeps startup and profile editing independent of external
    service availability.
    """
    defaults = studio_overrides(resolve_settings(profile=profile))
    comfy_url = str(defaults.get("comfy_url", "http://127.0.0.1:8188")).rstrip("/")
    reviewer_url = str(defaults.get("localdeploy_url", "http://127.0.0.1:8000/v1")).rstrip("/")
    endpoints = {
        "checkpoints": f"{comfy_url}/models/checkpoints",
        "diffusion_models": f"{comfy_url}/models/diffusion_models",
        "text_encoders": f"{comfy_url}/models/text_encoders",
        "vae": f"{comfy_url}/models/vae",
        "reviewer": f"{reviewer_url}/models",
    }
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        futures = {name: pool.submit(_json_get, url) for name, url in endpoints.items()}
        results = {name: future.result() for name, future in futures.items()}

    def string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return sorted(str(item) for item in value if isinstance(item, str))

    checkpoints = string_list(results["checkpoints"][0])
    diffusion_models = string_list(results["diffusion_models"][0])
    text_encoders = string_list(results["text_encoders"][0])
    vaes = string_list(results["vae"][0])
    reviewer_payload = results["reviewer"][0]
    reviewer_models = []
    if isinstance(reviewer_payload, dict) and isinstance(reviewer_payload.get("data"), list):
        reviewer_models = sorted(
            str(item["id"])
            for item in reviewer_payload["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
    qwen_ready = all(
        required in installed
        for required, installed in (
            ("qwen_image_2512_fp8_e4m3fn.safetensors", diffusion_models),
            ("qwen_2.5_vl_7b_fp8_scaled.safetensors", text_encoders),
            ("qwen_image_vae.safetensors", vaes),
        )
    )
    z_image_ready = all(
        required in installed
        for required, installed in (
            ("z_image_turbo_int8_convrot.safetensors", diffusion_models),
            ("qwen_3_4b_fp8_mixed.safetensors", text_encoders),
            ("ae.safetensors", vaes),
        )
    )
    comfy_ready = any(results[name][0] is not None for name in endpoints if name != "reviewer")
    return {
        "defaults": {
            key: value
            for key, value in defaults.items()
            if key in {
                "concept_backend",
                "checkpoint",
                "model",
                "spec_strategy",
                "concept_steps",
                "concept_cfg",
                "concept_candidates",
                "device_policy",
            }
            and value is not None
        },
        "checkpoints": checkpoints,
        "diffusion_models": diffusion_models,
        "review_models": reviewer_models,
        "qwen_image_2512_ready": qwen_ready,
        "z_image_turbo_ready": z_image_ready,
        "services": {
            "comfyui": {
                "ready": comfy_ready,
                "detail": (
                    results["checkpoints"][1]
                    if comfy_ready
                    else _service_unavailable_detail("start ComfyUI")
                ),
            },
            "reviewer": {
                "ready": reviewer_payload is not None,
                "detail": (
                    results["reviewer"][1]
                    if reviewer_payload is not None
                    else _service_unavailable_detail("start Ollama or LocalDeploy")
                ),
            },
        },
    }


def _services_needed(
    missing: list[dict[str, str]],
    description: str,
    profile: str,
    csrf: str,
    *,
    intended_use: str = "",
    must_have_features: str = "",
) -> str:
    """What to start, instead of a run that would fail on its first call.

    Deliberately not phrased as an error: nothing went wrong, a required
    local service simply is not running yet. The description is carried
    back into the form so retrying costs one click, not retyping.
    """
    rows = "".join(
        f'<li><strong>{html.escape(item["name"])}</strong> '
        f'<span class="badge needs">not running</span><br>'
        f'<span class=muted>{html.escape(item["url"])} did not answer.</span><br>'
        f'<code>{html.escape(item["remedy"])}</code></li>'
        for item in missing
    )
    plural = "service" if len(missing) == 1 else "services"
    return (
        '<section class="card hero"><h1>Start your local AI services first</h1>'
        f"<p>Text2Model Forge needs {len(missing)} local {plural} that {'is' if len(missing) == 1 else 'are'} "
        "not running yet. Nothing was created, so no failed run is left behind &mdash; "
        "start the below, then press Try again.</p>"
        f'<ul class="decisions">{rows}</ul>'
        '<p class=muted>The launcher can install and start everything for you: '
        '<code>.\\run.ps1 -AiStack qwen</code> on Windows, or '
        '<code>./run.sh --ai-stack qwen</code> on Linux and macOS. '
        'The <a href="/doctor">System page</a> shows every check in detail.</p>'
        '<form method=post action=/runs>'
        f'<input type=hidden name=csrf value="{csrf}">'
        f'<input type=hidden name=description value="{html.escape(description, quote=True)}">'
        f'<input type=hidden name=profile value="{html.escape(profile, quote=True)}">'
        f'<input type=hidden name=intended_use value="{html.escape(intended_use, quote=True)}">'
        f'<input type=hidden name=must_have_features value="{html.escape(must_have_features, quote=True)}">'
        '<button class=primary type=submit>Try again</button>'
        f'<a class=button href="/new?prompt={quote(description)}">Edit the description</a>'
        "</form></section>"
    )


def _hint(text: str) -> str:
    """A small "?" badge next to a label that reveals `text` on hover or
    keyboard focus. Pure CSS (see .hint in STYLE), so it works without the
    page's JS having loaded. Replaces the New-asset form's previous pattern
    of a permanent <p class=muted> paragraph under every field -- accurate
    but the reason the panel read as a wall of text before anyone touched a
    control."""
    return f'<span class=hint tabindex=0>?<span class=tip>{html.escape(text)}</span></span>'


def _new_form(csrf: str, preset_description: str = "") -> str:
    profiles = _available_profiles()
    recommended_profile = recommend_stack(detect_hardware()).profile
    default = (
        recommended_profile
        if recommended_profile in profiles
        else "simple"
        if "simple" in profiles
        else profiles[0]
    )
    defaults = studio_overrides(resolve_settings(profile=default))
    options = "".join(
        f'<option value="{html.escape(name)}"{" selected" if name == default else ""}>{html.escape(name)}</option>'
        for name in profiles
    )
    backend_default = html.escape(str(defaults.get("concept_backend", "auto")))
    strategy_default = html.escape(str(defaults.get("spec_strategy", "monolithic")))
    checkpoint_default = html.escape(str(defaults.get("checkpoint", "")))
    model_default = html.escape(str(defaults.get("model", "")))
    steps_default = html.escape(str(defaults.get("concept_steps", 30)))
    cfg_default = html.escape(str(defaults.get("concept_cfg", 6.0)))
    candidates_default = html.escape(str(defaults.get("concept_candidates", 3)))
    device_default = html.escape(str(defaults.get("device_policy", "prefer_gpu")))
    return (
        '<div class=crumb><a href=/>&larr; Projects</a></div><section class=create-shell>'
        '<div class="card hero"><p class=eyebrow>New project</p>'
        '<span class=visually-hidden>Describe one original asset</span><h1>What do you want to make?</h1>'
        '<p class=lede>Describe the object and the result you need. Include moving parts, required states, dimensions, or handedness only when they matter.</p>'
        '<form method=post action=/runs data-setup-options>'
        f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
        '<label for=description>Asset brief</label><textarea id=description name=description minlength=20 maxlength=12000 required '
        'placeholder="A weathered stone well with an iron crank, built as a static game prop...">'
        f'{html.escape(preset_description)}</textarea>'
        '<p class=field-hint>Use plain language. Studio compiles this into a typed production contract.</p>'
        '<div class=form-grid><div><label for=intended-use>Intended use <span class=muted>(optional)</span></label>'
        '<select id=intended-use name=intended_use><option value="">Not specified</option>'
        '<option>Game-ready asset</option><option>Animation or cinematic</option>'
        '<option>Visualization or prototype</option><option>Research or evaluation</option></select></div>'
        '<div><label for=must-have>Must-have features <span class=muted>(optional)</span></label>'
        '<input id=must-have name=must_have_features maxlength=1000 placeholder="For example: separate crank, clean silhouette"></div></div>'
        # Collapsed for "simple" (the point of that profile is exactly not
        # needing to see a checkpoint or device-policy field) and open for
        # anything else, since "advanced" and "8gb" both depend on values in
        # here. Still a plain <details>, so a simple-profile user who wants
        # one override can always click it open themselves -- this only
        # changes the default, not the access.
        f'<details class=options{" open" if default != "simple" else ""}>'
        '<summary>Generation and reviewer options</summary>'
        f'<label for=profile>Configuration profile</label><select id=profile name=profile>{options}</select>'
        f'<p class=muted>Leave any field blank to inherit the selected profile.{_hint("Installed model names appear as suggestions in the checkpoint and reviewer fields once local services are running.")}</p>'
        '<div class=option-grid>'
        # Costs quoted in the option text and hints below are wall-clock per
        # candidate at 768x1024 on an 8 GB RTX 3080, taken from a real
        # side-by-side run of one prompt through every backend -- not vendor
        # claims. D1 renders `concept_candidates` of these per iteration and
        # re-runs the whole set on rejection, so the per-image figure is the
        # one that decides whether a run finishes tonight.
        f'<div><label for=concept-backend>Text-to-2D backend{_hint("Which model renders D1 concept images. Qwen and Z-Image ignore the checkpoint field below; SDXL uses it.")}</label>'
        '<select id=concept-backend name=concept_backend>'
        f'<option value="">Profile default: {backend_default}</option>'
        '<option value=auto>Auto — Z-Image if installed, else Qwen, else SDXL</option>'
        '<option value=z_image_turbo>Z-Image Turbo — stylized, ~50 s/image (recommended)</option>'
        '<option value=qwen_image_2512>Qwen Image 2512 — best quality, ~10 min/image</option>'
        '<option value=sdxl>SDXL checkpoint — fastest, ~20 s/image</option>'
        '</select></div>'
        f'<div><label for=checkpoint>SDXL / custom checkpoint{_hint("Only read when the backend above is SDXL. Leave blank to use the profile default.")}</label>'
        f'<input id=checkpoint name=checkpoint list=installed-checkpoints maxlength=300 placeholder="Profile default: {checkpoint_default}">'
        '<datalist id=installed-checkpoints></datalist></div>'
        f'<div><label for=review-model>Qwen reviewer model{_hint("The vision model that compares D1 candidates and drives every later review gate. Leave blank to use the profile default.")}</label>'
        f'<input id=review-model name=model list=installed-review-models maxlength=300 placeholder="Profile default: {model_default}">'
        '<datalist id=installed-review-models></datalist></div>'
        f'<div><label for=spec-strategy>D0 spec strategy{_hint("How your description becomes the typed asset contract. Chunked suits a 7-8B local reviewer; Monolithic expects the qualified 27B model.")}</label>'
        '<select id=spec-strategy name=spec_strategy>'
        f'<option value="">Profile default: {strategy_default}</option>'
        '<option value=chunked>Chunked — best for 7–8B local models</option>'
        '<option value=monolithic>Monolithic — qualified 27B model</option>'
        '</select></div>'
        f'<div><label for=concept-steps>Concept steps{_hint("Diffusion steps per D1 candidate. More steps cost more time for usually sharper detail.")}</label>'
        f'<input id=concept-steps name=concept_steps type=number min=1 max=150 placeholder="Profile default: {steps_default}"></div>'
        f'<div><label for=concept-cfg>Concept CFG{_hint("How closely the render follows the prompt. Higher is more literal and less varied; lower allows more creative drift.")}</label>'
        f'<input id=concept-cfg name=concept_cfg type=number min=0.1 max=30 step=0.1 placeholder="Profile default: {cfg_default}"></div>'
        f'<div><label for=concept-candidates>Sequential candidate budget{_hint("How many D1 concepts to generate one after another before the reviewer compares the best few. More candidates cost more time but improve the odds of a keeper.")}</label>'
        f'<input id=concept-candidates name=concept_candidates type=number min=2 max=12 placeholder="Profile default: {candidates_default}"></div>'
        f'<div><label for=device-policy>Device policy{_hint("Whether inference may fall back to CPU when the GPU will not fit it. Prefer GPU allows a silent, much slower fallback; GPU compute only fails fast instead. Check the System page for this machine's recommendation.")}</label>'
        '<select id=device-policy name=device_policy>'
        f'<option value="">Profile default: {device_default}</option>'
        '<option value=gpu_compute_only>GPU compute only — requires live telemetry</option>'
        '<option value=prefer_gpu>Prefer GPU — CPU inference allowed</option>'
        '<option value=strict_device_only>Strict device only — experimental</option>'
        '</select></div>'
        '</div><div id=setup-service-status class=service-status role=status aria-live=polite>Checking local AI services and installed models...</div>'
        '</details><div class=hero-actions><button class=primary type=submit>Start build</button>'
        '<a class="button ghost" href=/>Cancel</a></div></form></div>'
        '<aside class="card create-aside"><p class=eyebrow>What happens next</p><h2>A reviewable process</h2>'
        '<ol><li>Studio turns the brief into a production contract.</li>'
        '<li>You choose or reject the visual concept.</li><li>Each approved result becomes input to the next stage.</li>'
        '<li>The final GLB and its provenance remain downloadable.</li></ol>'
        '<p class=muted>Every human gate binds the decision to exact evidence hashes.</p>'
        '<a href=/doctor>Check system readiness</a></aside></section>'
        '<script src="/static/studio.js" defer></script>'
    )


_CONCEPT_BACKENDS = {"auto", "qwen_image_2512", "qwen_image_edit_2511", "sdxl", "z_image_turbo"}
_SPEC_STRATEGIES = {"monolithic", "chunked"}
_DEVICE_POLICIES = {"prefer_gpu", "gpu_compute_only", "strict_device_only"}


def _new_run_overrides(values: dict[str, str], profile: str) -> dict[str, Any]:
    """Merge optional browser fields over one resolved profile.

    Empty fields intentionally do not become empty run settings; they mean
    "inherit" so switching profiles keeps working exactly as it did before
    the advanced controls were added.
    """
    result = studio_overrides(resolve_settings(profile=profile))
    backend = values.get("concept_backend", "").strip()
    if backend:
        if backend not in _CONCEPT_BACKENDS:
            raise ValueError(f"unknown text-to-2D backend: {backend}")
        result["concept_backend"] = backend
    strategy = values.get("spec_strategy", "").strip()
    if strategy:
        if strategy not in _SPEC_STRATEGIES:
            raise ValueError(f"unknown D0 spec strategy: {strategy}")
        result["spec_strategy"] = strategy
    device_policy = values.get("device_policy", "").strip()
    if device_policy:
        if device_policy not in _DEVICE_POLICIES:
            raise ValueError(f"unknown device policy: {device_policy}")
        result["device_policy"] = device_policy
    for key in ("checkpoint", "model"):
        value = values.get(key, "").strip()
        if value:
            result[key] = value
    raw_steps = values.get("concept_steps", "").strip()
    if raw_steps:
        try:
            steps = int(raw_steps)
        except ValueError as exc:
            raise ValueError("concept steps must be a whole number from 1 to 150") from exc
        if not 1 <= steps <= 150:
            raise ValueError("concept steps must be a whole number from 1 to 150")
        result["concept_steps"] = steps
    raw_cfg = values.get("concept_cfg", "").strip()
    if raw_cfg:
        try:
            cfg = float(raw_cfg)
        except ValueError as exc:
            raise ValueError("concept CFG must be a number greater than 0 and no more than 30") from exc
        if not 0 < cfg <= 30:
            raise ValueError("concept CFG must be a number greater than 0 and no more than 30")
        result["concept_cfg"] = cfg
    raw_candidates = values.get("concept_candidates", "").strip()
    if raw_candidates:
        try:
            candidates = int(raw_candidates)
        except ValueError as exc:
            raise ValueError("concept candidates must be a whole number from 2 to 12") from exc
        if not 2 <= candidates <= 12:
            raise ValueError("concept candidates must be a whole number from 2 to 12")
        result["concept_candidates"] = candidates
    return result


def _worker_report() -> list[dict[str, Any]]:
    """Live worker readiness, the same check `python -m text2model_forge workers` runs.

    /doctor used to list only the *names* bound in config.local.toml, which
    says nothing about whether any of them can actually run -- the one
    question the page exists to answer. Failures are reported per worker
    rather than raised: a single unreadable manifest must not blank the page.
    """
    try:
        manifests = load_manifests()
    except Exception as exc:
        return [{"worker_id": "(manifests)", "ready": False, "health_error": f"{type(exc).__name__}: {exc}"}]
    config = load_local_config()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            worker_id: pool.submit(
                preflight,
                manifest,
                command_prefix=(binding.command_prefix if (binding := worker_binding(config, worker_id)) else None),
                timeout_seconds=1.5,
            )
            for worker_id, manifest in manifests.items()
        }
        report = []
        for worker_id, future in futures.items():
            try:
                report.append(future.result())
            except Exception as exc:
                report.append(
                    {"worker_id": worker_id, "ready": False, "health_error": f"{type(exc).__name__}: {exc}"}
                )
    return sorted(report, key=lambda item: (item.get("ready") is not True, item["worker_id"]))



def _preflight_report():
    """Hardware + every cross-stage check, for the System page."""
    from .hardware import detect_hardware, recommend_stack
    from .preflight import run_preflight

    try:
        hardware, checks = run_preflight(profile="simple", deep=False)
        return hardware, checks, recommend_stack(hardware)
    except Exception as exc:  # a broken check must not blank the page
        from .hardware import HardwareProfile
        from .preflight import Check

        blank = HardwareProfile()
        return (
            blank,
            [Check(name="preflight", status="fail", detail=f"{type(exc).__name__}: {exc}")],
            recommend_stack(blank),
        )


_STATUS_BADGE = {"ok": "", "warn": " needs", "fail": " needs", "skip": ""}


def _preflight_html(hardware, checks, recommendation) -> str:
    failures = [check for check in checks if check.status == "fail"]
    rows = "".join(
        f'<li><span class="badge{_STATUS_BADGE.get(check.status, "")}">{html.escape(check.status)}</span> '
        f"<strong>{html.escape(check.name)}</strong><br>"
        f"<span class=muted>{html.escape(check.detail)}</span>"
        + (
            f'<br><span class=warning>fix: {html.escape(check.remedy)}</span>'
            if check.remedy and check.status in {"fail", "warn"}
            else ""
        )
        + "</li>"
        for check in checks
    )
    headline = (
        f'<p class=error>{len(failures)} assumption{"" if len(failures) == 1 else "s"} '
        "will fail this run. Each one below would otherwise surface several stages in.</p>"
        if failures
        else '<p class="good">Every cross-stage assumption holds on this machine.</p>'
    )
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in recommendation.reasons)
    return (
        '<h2 style="margin-top:18px">Preflight</h2>'
        f"{headline}"
        f'<ul class="decisions">{rows}</ul>'
        f'<details style="margin-top:12px"><summary>Recommended stack for this hardware '
        f"(profile {html.escape(recommendation.profile)})</summary>"
        f"<p><span class=badge>reviewer {html.escape(recommendation.reviewer_size)}</span> "
        f"<span class=badge>spec_strategy {html.escape(recommendation.spec_strategy)}</span> "
        f"<span class=badge>voxel_fraction {recommendation.voxel_fraction}</span></p>"
        f"<ul>{reasons}</ul></details>"
    )


def _doctor_shell() -> str:
    """What /doctor returns immediately, before any check has run.

    The checks themselves are unchanged and still take as long as they take;
    this only stops the browser from showing nothing at all while they do.
    """
    return (
        '<section class="card hero" id=doctor-shell><p class=eyebrow>System</p>'
        '<h1>Is this computer ready?</h1>'
        '<p id=doctor-status class=checking role=status aria-live=polite>Checking hardware, local services, '
        "and every cross-stage assumption&hellip;</p>"
        '<div class="bar indeterminate"><span></span></div>'
        "<p class=muted>Several checks wait on a local service that may not be running, "
        "so this usually takes a few seconds. You can keep using the rest of Studio.</p>"
        "</section>"
        '<script src="/static/doctor.js" defer></script>'
    )


def _doctor() -> str:
    defaults = studio_overrides(resolve_settings(profile="simple"))
    reviewer_url = str(defaults.get("localdeploy_url", "http://127.0.0.1:8000/v1")).rstrip("/")
    comfy_url = str(defaults.get("comfy_url", "http://127.0.0.1:8188")).rstrip("/")
    with ThreadPoolExecutor(max_workers=3) as pool:
        # Each probe blocks for up to two seconds and the worker preflight
        # blocks for longer; run them together so the page is not the sum of
        # every timeout on a machine with nothing running.
        localdeploy_future = pool.submit(_probe, f"{reviewer_url}/models")
        comfy_future = pool.submit(_probe, f"{comfy_url}/system_stats")
        workers_future = pool.submit(_worker_report)
        # The cross-stage assumption checks. Same report `text2model_forge doctor`
        # prints, because a mismatch that only shows up in the CLI is a
        # mismatch the browser user still runs into three stages later.
        preflight_future = pool.submit(_preflight_report)
        localdeploy, localdeploy_detail = localdeploy_future.result()
        comfy, comfy_detail = comfy_future.result()
        workers = workers_future.result()
        hardware, checks, recommendation = preflight_future.result()
    config = load_local_config()
    ready = sum(1 for item in workers if item.get("ready") is True)
    rows = "".join(
        f'<li><strong>{html.escape(str(item["worker_id"]))}</strong> '
        f'<span class="badge{"" if item.get("ready") else " needs"}">'
        f'{"ready" if item.get("ready") else "not ready"}</span> '
        f'<span class=muted>{html.escape(str(item.get("executable") or item.get("health_error") or item.get("declared_lifecycle") or ""))}</span></li>'
        for item in workers
    )
    failed_checks = [check for check in checks if check.status == "fail"]
    issues: list[str] = []
    if not localdeploy:
        issues.append(
            f'<article class=choice-card><h2>Start the reviewer</h2><p>{html.escape(localdeploy_detail)}</p>'
            '<code>.\\run.ps1 -AiStack qwen</code></article>'
        )
    if not comfy:
        issues.append(
            f'<article class=choice-card><h2>Start image generation</h2><p>{html.escape(comfy_detail)}</p>'
            '<code>.\\run.ps1 -AiStack qwen</code></article>'
        )
    if not config:
        issues.append(
            '<article class=choice-card><h2>Create local configuration</h2>'
            '<p>Copy <code>machine.example.toml</code> to <code>config.local.toml</code> and set local paths.</p>'
            '</article>'
        )
    if failed_checks:
        issues.append(
            f'<article class=choice-card><h2>Resolve production assumptions</h2><p>'
            f'{len(failed_checks)} cross-stage check{"s" if len(failed_checks) != 1 else ""} failed. '
            'Open the technical report below for the exact remedies.</p></article>'
        )
    ready_for_build = not issues and ready == len(workers)
    status_title = "Ready to create" if ready_for_build else "Setup needs attention"
    status_copy = (
        "The required local services and deterministic workers are available."
        if ready_for_build
        else "Resolve the items below before starting a production project. The offline tutorial remains available."
    )
    issue_grid = (
        f'<div class=choice-grid>{"".join(issues)}</div>'
        if issues
        else '<div class="notice success">No blocking setup issue was detected.</div>'
    )
    return (
        '<section class="card hero"><p class=eyebrow>System readiness</p>'
        f'<h1>{html.escape(status_title)}</h1><p class=lede>{html.escape(status_copy)}</p>'
        '<div class=health>'
        f'<span class="{"" if localdeploy else "down"}">Qwen reviewer ({html.escape(reviewer_url)}): {html.escape(localdeploy_detail)}</span>'
        f'<span class="{"" if comfy else "down"}">ComfyUI ({html.escape(comfy_url)}): {html.escape(comfy_detail)}</span>'
        f'<span class="{"" if config else "down"}">Text2Model Forge config: {"loaded" if config else "missing (copy machine.example.toml to config.local.toml)"}</span>'
        f'<span class="{"" if ready else "down"}">Workers ready: {ready}/{len(workers)}</span>'
        f'<span class="{"" if hardware.detected else "down"}">GPU: '
        f'{html.escape((hardware.primary.name if hardware.primary else "not detected"))}'
        f'{f" · {hardware.vram_total_gb} GB" if hardware.vram_total_gb else ""}</span></div>'
        f'<div class=hero-actions><a class="button primary" href=/new>Create an asset</a>'
        '<a class="button secondary" href=/>Open projects</a></div></section>'
        f'<section><div class=section-head><div><p class=eyebrow>Next actions</p>'
        f'<h2>{"Nothing to fix" if not issues else "What to fix"}</h2></div></div>{issue_grid}</section>'
        '<details class="card advanced-drawer"><summary>Technical readiness report</summary>'
        + _preflight_html(hardware, checks, recommendation)
        + '<h2>Deterministic worker preflight</h2>'
        f'<ul class="decisions">{rows or "<li>No worker manifests were found.</li>"}</ul>'
        '<p class=muted>Studio binds to loopback by default. Qwen proposes structured decisions; '
        'it never executes code or edits artifacts.</p></details>'
    )


def _golden_dashboard(store: StudioStore) -> str:
    from .golden import load_corpus
    from text2model_forge.paths import resource_root

    corpus = load_corpus(resource_root() / "golden" / "static-props.json")
    runs_by_description: dict[str, list] = {}
    for run in store.list():
        runs_by_description.setdefault(run.description.strip(), []).append(run)
    cards: list[str] = []
    attempted = completed = 0
    for case in corpus.cases:
        matches = runs_by_description.get(case.prompt.strip(), [])
        run = matches[0] if matches else None
        if run is not None:
            attempted += 1
            completed += int(run.state == "completed")
            progress = _run_progress(run)
            action = f'<a class=button href="/run/{quote(run.run_id)}">Open latest run</a>'
            state = html.escape(run.state)
        else:
            progress = 0
            action = f'<a class="button primary" href="/new?prompt={quote(case.prompt)}">Start this case</a>'
            state = "not attempted"
        features = "".join(f"<li>{html.escape(item)}</li>" for item in case.required_features)
        cards.append(
            '<section class="card run-card">'
            f'<h2>{html.escape(case.case_id)}</h2><p><span class=badge>{html.escape(case.category)}</span> '
            f'<span class=badge>{state}</span></p><p>{html.escape(case.prompt)}</p>'
            f'<ul>{features}</ul><div class=progress-label><span>Run progress</span><span>{progress:.0%}</span></div>'
            f'{_progress_bar(progress, label=case.case_id + " corpus progress")}{action}</section>'
        )
    overall = completed / corpus.required_attempts
    return (
        '<section class="card hero"><h1>Live 8 GB static-prop qualification</h1>'
        f'<p>{attempted}/{corpus.required_attempts} attempted; {completed}/{corpus.required_attempts} completed. '
        f'Publication threshold: at least {corpus.minimum_passing_cases} human-reviewed passes after every case is attempted.</p>'
        f'<div class=progress-label><span>Completed live runs</span><span>{overall:.0%}</span></div>'
        f'{_progress_bar(overall, label="Golden corpus completion")} '
        '<p class=muted>Completion alone is not a pass. Export the human assessment report and run '
        '<code>text2model-forge golden evaluate</code>; the evaluator verifies the stored run evidence and fails closed.</p>'
        '</section><div class=grid style="margin-top:16px">' + "".join(cards) + "</div>"
    )
class StudioServer(NamedTuple):
    """A constructed but not-yet-running Studio server, plus the pieces a
    caller needs to drive or inspect it. Split out of serve() so the HTTP
    layer -- routing, CSRF, form parsing, error rendering -- can be tested
    against a real loopback server instead of only by reading it."""

    server: ThreadingHTTPServer
    store: StudioStore
    coordinator: StudioCoordinator
    csrf: str
    recovered: list[str]

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"


def build_server(
    workspace: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    allow_non_loopback: bool = False,
    coordinator_factory=None,
) -> StudioServer:
    """Construct the Studio HTTP server without running it.

    Pass port=0 for an ephemeral port. `coordinator_factory` takes the
    server's own StudioStore and returns a StudioCoordinator, so a test can
    supply fake Qwen/ComfyUI/worker providers. It deliberately receives the
    store rather than accepting a pre-built coordinator: two StudioStore
    instances over one directory hold independent locks, so a coordinator
    built on a different store than the request handlers use would race on
    run.json writes.
    """
    if host not in {"127.0.0.1", "::1", "localhost"} and not allow_non_loopback:
        raise ValueError(
            "Text2Model Forge Studio may bind only to a loopback address unless "
            "allow_non_loopback=True is explicitly set"
        )
    store = StudioStore(workspace)
    recovered = store.recover_interrupted_runs()
    coordinator = (coordinator_factory or StudioCoordinator)(store)
    application = StudioApplication(store, coordinator)
    csrf = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'",
            )
            self.end_headers()

        def page(self, title: str, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = _page(title, body)
            self._headers(status, "text/html; charset=utf-8", len(payload))
            self.wfile.write(payload)

        def redirect(self, path: str) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", path)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/":
                    show_archived = parse_qs(parsed.query).get("archived", [""])[0] == "1"
                    self.page(
                        "Text2Model Forge Studio",
                        _dashboard(store, csrf, show_archived=show_archived),
                    )
                elif path == "/new":
                    preset = parse_qs(parsed.query).get("prompt", [""])[0]
                    self.page("New asset", _new_form(csrf, preset))
                elif path == "/golden":
                    self.page("Golden corpus", _golden_dashboard(store))
                elif path == "/doctor":
                    self.page("System", _doctor_shell())
                elif path == "/favicon.ico":
                    self._headers(HTTPStatus.OK, "image/svg+xml", len(_FAVICON))
                    self.wfile.write(_FAVICON)
                elif path.startswith("/run/"):
                    parts = path.split("/")
                    if len(parts) >= 5 and parts[3] == "stage":
                        run_id = unquote(parts[2])
                        stage_id = unquote(parts[4])
                        self.page("Stage detail", _stage_page(store, run_id, stage_id))
                    else:
                        run_id = unquote(parts[2])
                        self.page("Production run", _run_page(store, coordinator, run_id, csrf))
                elif path == "/static/studio.js":
                    payload = STUDIO_JS.encode("utf-8")
                    self._headers(HTTPStatus.OK, "text/javascript; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path == "/static/glb-viewer.js":
                    payload = GLB_VIEWER_JS.encode("utf-8")
                    self._headers(HTTPStatus.OK, "text/javascript; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path == "/static/active-runs.js":
                    payload = ACTIVE_RUNS_JS.encode("utf-8")
                    self._headers(HTTPStatus.OK, "text/javascript; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path == "/static/doctor.js":
                    payload = DOCTOR_JS.encode("utf-8")
                    self._headers(HTTPStatus.OK, "text/javascript; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path == "/api/doctor":
                    payload = _doctor().encode("utf-8")
                    self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path in {"/api/setup/options", "/api/v1/system/setup-options"}:
                    profile = parse_qs(parsed.query).get("profile", ["simple"])[0]
                    if profile not in _available_profiles():
                        raise ValueError(f"unknown configuration profile: {profile}")
                    payload = json.dumps(_setup_options(profile)).encode("utf-8")
                    self._headers(HTTPStatus.OK, "application/json; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path == "/api/active-runs":
                    payload = json.dumps(_active_runs_payload(store)).encode("utf-8")
                    self._headers(HTTPStatus.OK, "application/json; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path == "/api/v1/projects":
                    payload = json.dumps(
                        [
                            {
                                "run_id": run.run_id,
                                "title": run.title,
                                "state": run.state,
                                "run_mode": run.run_mode,
                                "current_stage": run.current_stage,
                                "progress": _run_progress(run),
                                "updated_at": run.updated_at.isoformat(),
                            }
                            for run in application.list_projects(include_archived=True)
                        ],
                        indent=2,
                    ).encode("utf-8")
                    self._headers(HTTPStatus.OK, "application/json; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path.startswith("/api/v1/projects/"):
                    remainder = path.removeprefix("/api/v1/projects/")
                    if remainder.endswith("/events"):
                        run_id = unquote(remainder[: -len("/events")].rstrip("/"))
                        application.get_project(run_id)
                        payload = json.dumps(store.read_events(run_id), indent=2).encode("utf-8")
                    else:
                        run_id = unquote(remainder.rstrip("/"))
                        payload = application.get_project(run_id).model_dump_json(indent=2).encode("utf-8")
                    self._headers(HTTPStatus.OK, "application/json; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path.startswith("/api/run/"):
                    run_id = unquote(path.split("/", 3)[3])
                    payload = store.load(run_id).model_dump_json(indent=2).encode("utf-8")
                    self._headers(HTTPStatus.OK, "application/json; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif path.startswith("/artifact/"):
                    parts = path.split("/", 3)
                    if len(parts) != 4:
                        raise FileNotFoundError("artifact path is incomplete")
                    run_id = unquote(parts[2])
                    relative = unquote(parts[3])
                    target = store.artifact_path(run_id, relative)
                    payload = target.read_bytes()
                    content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                    self._headers(HTTPStatus.OK, content_type, len(payload))
                    self.wfile.write(payload)
                else:
                    self.page("Not found", "<h1>Not found</h1>", HTTPStatus.NOT_FOUND)
            except (FileNotFoundError, ValueError) as exc:
                self.page("Not found", f'<section class=card><h1 class=error>Error</h1><pre>{html.escape(str(exc))}</pre></section>', HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                path = urlparse(self.path).path
                # A file upload cannot go through _form(): that parser reads
                # the whole body as one urlencoded blob and would consume it
                # from the socket before this branch got a chance to. Every
                # other route's body is small and urlencoded, so this is the
                # only path that needs to read the request itself.
                if path.startswith("/run/") and path.endswith("/upload-image"):
                    run_id = unquote(path.split("/")[2])
                    fields = _multipart_form(self, max_bytes=21 * 1024 * 1024)
                    if not secrets.compare_digest(str(fields.get("csrf", "")), csrf):
                        raise ValueError("invalid form token")
                    upload = fields.get("image")
                    if not isinstance(upload, tuple) or not upload[0]:
                        raise ValueError("choose an image file to upload")
                    filename, data = upload
                    if not coordinator.submit_manual_image_upload(run_id, data, filename):
                        raise ValueError("another Studio job is already running for this asset")
                    self.redirect("/run/" + quote(run_id))
                    return
                values = _form(self)
                if not secrets.compare_digest(values.get("csrf", ""), csrf):
                    raise ValueError("invalid form token")
                if path == "/demo":
                    run_id = "tutorial-" + _slug().removeprefix("asset-")
                    application.create_tutorial(run_id)
                    self.redirect("/run/" + quote(run_id))
                elif path == "/runs":
                    description = values.get("description", "").strip()
                    if len(description) < 20:
                        raise ValueError("description must contain at least 20 characters")
                    if len(description) > 12000:
                        raise ValueError("description must contain no more than 12000 characters")
                    intended_use = values.get("intended_use", "").strip()
                    allowed_uses = {
                        "",
                        "Game-ready asset",
                        "Animation or cinematic",
                        "Visualization or prototype",
                        "Research or evaluation",
                    }
                    if intended_use not in allowed_uses:
                        raise ValueError("unknown intended use")
                    must_have = values.get("must_have_features", "").strip()
                    if len(must_have) > 1000:
                        raise ValueError("must-have features must contain no more than 1000 characters")
                    profile = values.get("profile", "simple").strip() or "simple"
                    # resolve_settings() turns this straight into a
                    # profiles/<name>.toml path, so accept only a profile the
                    # selector actually offers rather than any string a form
                    # post happens to carry.
                    if profile not in _available_profiles():
                        raise ValueError(f"unknown configuration profile: {profile}")
                    overrides = {
                        **_new_run_overrides(values, profile),
                        "profile": profile,
                        "intended_use": intended_use or None,
                        "must_have_features": must_have or None,
                    }
                    # Refuse before creating anything. A run started without
                    # its reviewer dies on D0's first call and leaves a
                    # permanently failed run behind that never produced a
                    # single piece of evidence -- the failure the user hits
                    # is not the pipeline's, it is a setup step nobody was
                    # told about. Say so here, keep their description, and
                    # create nothing.
                    missing = coordinator.missing_services(overrides)
                    if missing:
                        self.page(
                            "Start your local AI services",
                            _services_needed(
                                missing,
                                description,
                                profile,
                                csrf,
                                intended_use=intended_use,
                                must_have_features=must_have,
                            ),
                        )
                        return
                    run_id = _slug()
                    application.create_project(run_id, description, overrides)
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/decision"):
                    run_id = unquote(path.split("/")[2])
                    overrides_raw = values.get("overrides", "").strip()
                    overrides = None
                    if overrides_raw:
                        try:
                            overrides = json.loads(overrides_raw)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Overrides must be valid JSON (for example {{\"seed\": 42}}): {exc.msg}"
                            ) from exc
                        if not isinstance(overrides, dict):
                            raise ValueError(
                                'Overrides must be a JSON object, for example {"seed": 42}.'
                            )
                    application.decide(
                        run_id,
                        values["stage_id"],
                        values["decision"],
                        values.get("comment", ""),
                        values.get("selected_evidence_id") or None,
                        overrides=overrides,
                        target_stage_id=values.get("target_stage_id") or None,
                        assisted_by_review_id=values.get("assisted_by_review_id") or None,
                    )
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/qwen-image"):
                    run_id = unquote(path.split("/")[2])
                    prompt = values.get("prompt", "").strip()
                    raw_seed = values.get("seed", "").strip()
                    try:
                        seed = int(raw_seed) if raw_seed else None
                    except ValueError as exc:
                        raise ValueError("seed must be a whole non-negative number") from exc
                    if not coordinator.submit_manual_qwen_image(run_id, prompt, seed=seed):
                        raise ValueError("another Studio job is already running for this asset")
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/stop"):
                    run_id = unquote(path.split("/")[2])
                    accepted, message = coordinator.stop(run_id)
                    if not accepted:
                        raise ValueError(message)
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/gpu-free"):
                    run_id = unquote(path.split("/")[2])
                    released, message = coordinator.release_comfy_memory(run_id)
                    if not released:
                        raise ValueError(message)
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/resume"):
                    run_id = unquote(path.split("/")[2])
                    application.resume(run_id)
                    self.redirect("/run/" + quote(run_id))
                elif path.startswith("/run/") and path.endswith("/archive"):
                    run_id = unquote(path.split("/")[2])
                    application.set_archived(run_id, True)
                    self.redirect("/")
                elif path.startswith("/run/") and path.endswith("/unarchive"):
                    run_id = unquote(path.split("/")[2])
                    application.set_archived(run_id, False)
                    self.redirect("/run/" + quote(run_id))
                else:
                    self.page("Not found", "<h1>Not found</h1>", HTTPStatus.NOT_FOUND)
            except StudioConflictError as exc:
                self.page(
                    "Project changed",
                    '<section class=card><h1>Reload before applying that decision</h1>'
                    f'<p>{html.escape(str(exc))}</p><a class="button primary" href="{html.escape(self.path, quote=True)}">'
                    'Reload project</a></section>',
                    HTTPStatus.CONFLICT,
                )
            except (KeyError, FileNotFoundError, ValueError) as exc:
                self.page(
                    "Text2Model Forge Studio error",
                    f'<section class=card><h1 class=error>Could not apply that action</h1><pre>{html.escape(str(exc))}</pre><a class=button href="/">Back to runs</a></section>',
                    HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"Text2Model Forge Studio {self.address_string()}: {format % args}")

    return StudioServer(
        server=ThreadingHTTPServer((host, port), Handler),
        store=store,
        coordinator=coordinator,
        csrf=csrf,
        recovered=recovered,
    )


def serve(
    workspace: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = False,
    allow_non_loopback: bool = False,
) -> None:
    studio = build_server(
        workspace,
        host=host,
        port=port,
        allow_non_loopback=allow_non_loopback,
    )
    if studio.recovered:
        print(f"Text2Model Forge Studio recovered interrupted runs: {', '.join(studio.recovered)}")
    print(f"Text2Model Forge Studio: {studio.url}")
    if open_browser:
        webbrowser.open(studio.url + "/new")
    try:
        studio.server.serve_forever()
    finally:
        studio.coordinator.close()
        studio.server.server_close()
