"""Static design-system assets for the local Studio web application."""

MODERN_STYLE = r"""
:root {
  color-scheme: dark;
  --bg: #0a0e11;
  --surface: #11181c;
  --surface-raised: #172126;
  --surface-soft: #1d292f;
  --surface-sunken: #080c0e;
  --border: #314149;
  --border-strong: #4a5d66;
  --text: #f4efe7;
  --text-soft: #c7d0d2;
  --muted: #91a0a5;
  --accent: #ec6f3d;
  --accent-hover: #f48050;
  --accent-soft: #46261b;
  --blue: #65a6c3;
  --blue-soft: #1a3440;
  --success: #6fc485;
  --success-soft: #183523;
  --warning: #e0b34f;
  --warning-soft: #3b2d14;
  --danger: #ef756c;
  --danger-soft: #401f1d;
  --shadow: 0 18px 48px rgba(0, 0, 0, .28);
  --radius-sm: 8px;
  --radius: 14px;
  --radius-lg: 20px;
  --content: 1480px;
}

* { box-sizing: border-box; }
html { min-width: 0; background: var(--bg); }
body {
  min-width: 0;
  margin: 0;
  overflow-x: hidden;
  background:
    radial-gradient(circle at 12% -12%, rgba(65, 99, 110, .25), transparent 34rem),
    linear-gradient(180deg, #0b1114 0, var(--bg) 32rem);
  color: var(--text);
  font: 16px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  text-rendering: optimizeLegibility;
}

a { color: #b9d9e7; text-underline-offset: 3px; }
a:hover { color: #e3f4fb; }
button, input, select, textarea { font: inherit; }
button, .button, summary, select, input[type="radio"], input[type="file"] { touch-action: manipulation; }
:focus-visible { outline: 3px solid #f2a06f; outline-offset: 3px; }

.skip-link {
  position: fixed;
  z-index: 100;
  top: 8px;
  left: 8px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  background: var(--text);
  color: var(--bg);
  transform: translateY(-160%);
}
.skip-link:focus { transform: translateY(0); }

.app-header {
  position: sticky;
  z-index: 20;
  top: 0;
  border-bottom: 1px solid rgba(74, 93, 102, .72);
  background: rgba(10, 15, 18, .92);
  backdrop-filter: blur(18px);
}
.header-inner {
  display: flex;
  align-items: center;
  width: min(100%, var(--content));
  min-height: 66px;
  margin: auto;
  padding: 0 24px;
  gap: 26px;
}
.brand {
  display: inline-flex;
  flex: none;
  align-items: center;
  min-height: 44px;
  gap: 11px;
  color: var(--text);
  font-weight: 760;
  letter-spacing: -.015em;
  text-decoration: none;
}
.brand .logo { flex: none; }
.brand-kicker {
  margin-left: 2px;
  padding: 3px 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 680;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.desktop-nav { display: flex; align-items: center; gap: 4px; }
.desktop-nav > a, .nav-popover > summary {
  display: inline-flex;
  align-items: center;
  min-height: 44px;
  padding: 0 12px;
  border-radius: var(--radius-sm);
  color: var(--text-soft);
  font-size: 14px;
  font-weight: 620;
  text-decoration: none;
  cursor: pointer;
}
.desktop-nav > a:hover, .desktop-nav > a[aria-current="page"], .nav-popover > summary:hover {
  background: var(--surface-soft);
  color: var(--text);
}
.nav-popover { position: relative; }
.nav-popover > summary { list-style: none; }
.nav-popover > summary::-webkit-details-marker { display: none; }
.nav-popover-menu {
  position: absolute;
  top: calc(100% + 6px);
  right: 0;
  width: 230px;
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface-raised);
  box-shadow: var(--shadow);
}
.nav-popover-menu a {
  display: block;
  min-height: 44px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  color: var(--text-soft);
  text-decoration: none;
}
.nav-popover-menu a:hover { background: var(--surface-soft); color: var(--text); }
.mobile-nav { display: none; margin-left: auto; position: relative; }
.mobile-nav > summary {
  display: flex;
  align-items: center;
  min-height: 44px;
  padding: 0 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  list-style: none;
  cursor: pointer;
}
.mobile-nav > summary::-webkit-details-marker { display: none; }
.mobile-nav .nav-popover-menu { right: 0; }
.active-runs { margin-left: auto; display: flex; gap: 7px; min-width: 0; }
.active-run-chip {
  display: flex;
  align-items: center;
  max-width: 230px;
  min-height: 36px;
  padding: 6px 10px;
  gap: 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  color: var(--text-soft);
  font-size: 12px;
  text-decoration: none;
}
.active-run-chip .title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.active-run-chip .dot { width: 8px; height: 8px; flex: none; border-radius: 50%; background: var(--blue); }
.active-run-chip.awaiting_review .dot { background: var(--warning); }
.active-run-chip small { color: var(--muted); white-space: nowrap; }

.wrap { width: min(100%, var(--content)); margin: auto; padding: 28px 24px 64px; }
h1, h2, h3 { margin: 0; line-height: 1.18; letter-spacing: -.025em; }
h1 { font-size: clamp(30px, 3vw, 48px); }
h2 { font-size: clamp(21px, 1.6vw, 27px); }
h3 { font-size: 17px; }
p { margin: 9px 0; }
.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  margin: 0 0 10px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 760;
  letter-spacing: .11em;
  text-transform: uppercase;
}
.lede { max-width: 70ch; color: var(--text-soft); font-size: clamp(17px, 1.5vw, 20px); }
.muted { color: var(--muted); }
.good { color: #a7e6b7; }
.warning { color: #f1cd76; }
.error { color: #ff9b94; }
.visually-hidden {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  margin: -1px !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  white-space: nowrap !important;
  border: 0 !important;
}

.card {
  min-width: 0;
  padding: 22px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: linear-gradient(145deg, rgba(24, 34, 39, .98), rgba(16, 23, 27, .98));
  box-shadow: 0 10px 28px rgba(0, 0, 0, .14);
}
.card > :first-child { margin-top: 0; }
.hero { padding: clamp(24px, 4vw, 46px); border-radius: var(--radius-lg); }
.hero-actions, .actions, .prompt-tools { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
.hero-actions { margin-top: 22px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 310px), 1fr)); gap: 16px; }
.dashboard-shell { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 20px; align-items: start; }
.dashboard-main { min-width: 0; }
.dashboard-aside { position: sticky; top: 90px; display: grid; gap: 16px; }
.project-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.section-head { display: flex; justify-content: space-between; align-items: end; margin: 28px 0 14px; gap: 18px; }
.section-head p { color: var(--muted); }
.empty-state { display: grid; place-items: center; min-height: 320px; text-align: center; }
.empty-state > div { max-width: 560px; }
.choice-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 22px; }
.choice-card {
  min-width: 0;
  padding: 20px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}
.choice-card .choice-icon { display: grid; place-items: center; width: 42px; height: 42px; margin-bottom: 18px; border-radius: 12px; background: var(--accent-soft); color: #ffb18a; font-size: 21px; }
.choice-card p { color: var(--muted); }

button, .button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 44px;
  min-width: 44px;
  margin: 0;
  padding: 10px 16px;
  gap: 8px;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  background: #3a4d56;
  color: white;
  font-weight: 700;
  line-height: 1.15;
  text-align: center;
  text-decoration: none;
  cursor: pointer;
}
button:hover, .button:hover { filter: brightness(1.09); color: white; }
button:disabled { cursor: not-allowed; opacity: .5; }
.primary { background: var(--accent) !important; }
.secondary { border-color: var(--border-strong); background: var(--surface-soft) !important; }
.ghost { border-color: var(--border); background: transparent !important; color: var(--text-soft); }
.danger, .reject { background: #9b3f39 !important; }
.memory { background: #765d25 !important; }
.button.block { width: 100%; }

label, legend { color: var(--text-soft); font-size: 14px; font-weight: 680; }
label { display: block; margin: 16px 0 7px; }
fieldset { min-width: 0; margin: 18px 0; padding: 0; border: 0; }
legend { margin-bottom: 8px; }
textarea, input, select {
  width: 100%;
  min-width: 0;
  min-height: 46px;
  padding: 11px 13px;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--surface-sunken);
  color: var(--text);
}
textarea { min-height: 145px; resize: vertical; }
textarea::placeholder, input::placeholder { color: #718086; opacity: 1; }
.field-hint { margin-top: 6px; color: var(--muted); font-size: 13px; }
.form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 18px; }
.create-shell { display: grid; grid-template-columns: minmax(0, 780px) minmax(280px, 1fr); gap: 22px; align-items: start; }
.create-aside { position: sticky; top: 90px; }
.options { margin-top: 22px; padding: 14px; border: 1px solid var(--border); border-radius: var(--radius); background: rgba(7, 12, 14, .55); }
.options > summary { display: flex; align-items: center; min-height: 44px; cursor: pointer; font-weight: 730; }
.option-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 16px; }
.hint { position: relative; display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; margin-left: 5px; border-radius: 50%; background: var(--surface-soft); color: var(--muted); font-size: 12px; cursor: help; }
.hint .tip { visibility: hidden; position: absolute; z-index: 30; bottom: calc(100% + 8px); left: 0; width: min(270px, 72vw); padding: 10px 12px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: #0c1215; color: var(--text-soft); box-shadow: var(--shadow); opacity: 0; }
.hint:hover .tip, .hint:focus-visible .tip { visibility: visible; opacity: 1; }
.service-status, .status-line, .notice {
  margin-top: 14px;
  padding: 12px 14px;
  border-left: 4px solid var(--blue);
  border-radius: var(--radius-sm);
  background: var(--surface-sunken);
  color: var(--text-soft);
}
.service-status.good, .notice.success { border-left-color: var(--success); }
.service-status.warning, .notice.warning { border-left-color: var(--warning); }
.notice.tutorial { border-left-color: var(--blue); background: var(--blue-soft); }

.badge {
  display: inline-flex;
  align-items: center;
  min-height: 25px;
  padding: 3px 8px;
  border-radius: 999px;
  background: #26343a;
  color: #d0dbde;
  font-size: 12px;
  font-weight: 680;
  line-height: 1.15;
}
.badge.needs, .recommended { background: var(--warning-soft) !important; color: #f6d77d !important; }
.badge.gate { background: #342a40; color: #ddc9ed; }
.badge.na { color: var(--muted); }
.metrics, .prompt-meta, .health { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }

.run-header { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 22px; align-items: start; }
.run-summary { min-width: 0; }
.run-actions { display: flex; gap: 8px; }
.run-workspace { display: grid; grid-template-columns: 210px minmax(0, 1fr) 340px; gap: 18px; margin-top: 18px; align-items: start; }
.stage-rail, .review-panel { position: sticky; top: 88px; min-width: 0; }
.review-panel { display: grid; gap: 14px; }
.workspace-main { min-width: 0; }
.workspace-main > * + * { margin-top: 14px; }
.workspace-main .section-head { margin: 0 0 14px; }
.completion { margin-top: 18px; }
.phase-list { display: grid; gap: 5px; }
.phase-link {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr);
  min-height: 52px;
  padding: 8px;
  gap: 8px;
  border-radius: var(--radius-sm);
  color: var(--text-soft);
  text-decoration: none;
}
.phase-link:hover, .phase-link.active { background: var(--surface-soft); color: var(--text); }
.phase-dot { display: grid; place-items: center; width: 26px; height: 26px; border: 1px solid var(--border-strong); border-radius: 50%; font-size: 11px; font-weight: 750; }
.phase-link.approved .phase-dot { border-color: var(--success); background: var(--success-soft); }
.phase-link.awaiting_review .phase-dot { border-color: var(--warning); background: var(--warning-soft); }
.phase-link small { display: block; color: var(--muted); }
.timeline { display: flex; max-width: 100%; margin: 18px 0; padding: 2px 2px 10px; gap: 7px; overflow-x: auto; overscroll-behavior-x: contain; scrollbar-width: thin; }
.timeline .stage { flex: 0 0 125px; min-width: 0; min-height: 104px; padding: 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-sunken); color: inherit; text-decoration: none; }
.timeline .stage strong, .timeline .stage small { display: block; }
.timeline .stage small { min-height: 38px; color: var(--muted); }
.timeline .stage.active { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.timeline .stage.approved { border-color: #376948; background: var(--success-soft); }
.timeline .stage.awaiting_review { border-color: #8f7132; background: var(--warning-soft); }
.timeline .stage.failed, .timeline .stage.rejected { border-color: #7b3a36; background: var(--danger-soft); }
.timeline .stage.skipped { opacity: .58; border-style: dashed; }
.bar { height: 6px; margin-top: 10px; overflow: hidden; border-radius: 999px; background: #070a0c; }
.bar.overall { height: 10px; }
.bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), #f3a266); transition: width .3s ease; }
.bar.gpu span { background: linear-gradient(90deg, var(--blue), #86cce6); }
.progress-label { display: flex; justify-content: space-between; gap: 10px; margin-top: 12px; color: var(--muted); font-size: 13px; }

.evidence-heading { margin: 26px 0 12px; }
.attempt { display: flex; align-items: center; gap: 9px; margin: 22px 0 10px; }
.attempt h3 { margin: 0; }
.evidence-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.evidence { display: flex; min-width: 0; flex-direction: column; }
.evidence.is-candidate { padding: 14px; }
.evidence img { display: block; width: 100%; max-height: 640px; aspect-ratio: 4 / 3; object-fit: contain; border: 1px solid var(--border); border-radius: 10px; background: #05080a; }
.evidence-select { display: flex; align-items: center; min-height: 48px; margin-top: 10px; padding: 10px 12px; gap: 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-sunken); cursor: pointer; }
.evidence-select:has(input:checked) { border-color: var(--accent); background: var(--accent-soft); }
.evidence-select input { width: 22px; min-height: 22px; margin: 0; accent-color: var(--accent); }
.glb-preview { display: block; width: 100%; height: min(58vh, 620px); min-height: 380px; border: 1px solid var(--border); border-radius: 10px; background: radial-gradient(circle, #283940, #070a0c 72%); cursor: grab; touch-action: none; }
.glb-preview:active { cursor: grabbing; }
.viewer-note { color: var(--muted); font-size: 13px; }
.inspection-drawer, .advanced-drawer { margin-top: 12px; border-top: 1px solid var(--border); }
.inspection-drawer > summary, .advanced-drawer > summary { display: flex; align-items: center; min-height: 44px; color: var(--text-soft); cursor: pointer; }
pre, code { max-width: 100%; border: 1px solid #26343a; border-radius: 7px; background: var(--surface-sunken); }
pre { padding: 12px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
code { padding: 2px 5px; }

.review-panel .card { padding: 18px; }
.review-panel .actions { display: grid; }
.review-panel .actions button, .review-panel .button { width: 100%; }
.decision-primary { display: grid; gap: 10px; margin-top: 16px; }
.decision-more { margin-top: 12px; border-top: 1px solid var(--border); }
.decision-more > summary { display: flex; align-items: center; min-height: 44px; cursor: pointer; color: var(--text-soft); font-weight: 700; }
.decision-more .actions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.console { margin-top: 18px; border-color: #45606c; }
.console-head { display: flex; justify-content: space-between; gap: 12px; }
.control-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }
.control-action { min-width: 0; padding: 13px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-sunken); }
.control-action p { color: var(--muted); font-size: 13px; }
.events { max-height: 360px; overflow: auto; }
.event { margin: 6px 0; padding: 7px 0 7px 12px; border-left: 2px solid var(--border); overflow-wrap: anywhere; }
.decisions { margin: 8px 0 0; padding: 0; list-style: none; }
.decisions li { margin: 7px 0; padding: 8px 0 8px 12px; border-left: 3px solid var(--border); }
.decisions .approve { border-color: var(--success); }
.decisions .reject, .decisions .rollback { border-color: var(--danger); }
.decisions .edit, .decisions .retry { border-color: var(--blue); }
.crumb { display: flex; flex-wrap: wrap; align-items: center; margin-bottom: 14px; gap: 9px; }
.run-card { display: flex; flex-direction: column; min-height: 260px; }
.project-thumb { display: block; width: calc(100% + 44px); height: 190px; margin: -22px -22px 18px; border: 0; border-bottom: 1px solid var(--border); border-radius: var(--radius) var(--radius) 0 0; background: var(--surface-sunken); object-fit: cover; }
.project-thumb-empty { display: grid; place-items: center; color: var(--border-strong); font-size: 46px; }
.run-card h2 a { color: var(--text); text-decoration: none; }
.run-card .card-action { margin-top: auto; padding-top: 16px; }
.run-card .bar { margin-bottom: 5px; }
.health { gap: 8px; }
.health span { padding: 7px 10px; border-radius: var(--radius-sm); background: #26343a; }
.health .down { background: var(--danger-soft); color: #ffb0a9; }
.checking { color: var(--muted); animation: pulse 1.5s ease-in-out infinite; }
.bar.indeterminate span { width: 38%; animation: slide 1.45s ease-in-out infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .55; } }
@keyframes slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(300%); } }

@media (max-width: 1180px) {
  .project-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .run-workspace { grid-template-columns: 190px minmax(0, 1fr); }
  .review-panel { position: static; grid-column: 2; }
  .control-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .active-runs { display: none; }
}

@media (max-width: 820px) {
  .header-inner { min-height: 60px; padding: 0 14px; gap: 12px; }
  .desktop-nav { display: none; }
  .mobile-nav { display: block; }
  .brand-kicker { display: none; }
  .wrap { padding: 18px 14px 88px; }
  .dashboard-shell, .create-shell, .run-workspace { display: block; }
  .dashboard-aside, .create-aside, .stage-rail, .review-panel { position: static; margin-top: 16px; }
  .project-grid, .choice-grid, .evidence-grid, .form-grid, .option-grid { grid-template-columns: minmax(0, 1fr); }
  .hint .tip { right: 0; left: auto; width: min(270px, calc(100vw - 40px)); }
  .project-grid { gap: 12px; }
  .hero { padding: 23px 20px; }
  .card { padding: 18px; }
  .project-thumb { width: calc(100% + 36px); margin: -18px -18px 16px; }
  .run-header { display: block; }
  .run-actions { margin-top: 14px; }
  .stage-rail .phase-list { display: flex; max-width: 100%; overflow-x: auto; }
  .phase-link { flex: 0 0 150px; }
  .review-panel { margin-top: 14px; }
  .decision-primary {
    position: sticky;
    z-index: 12;
    bottom: 0;
    margin: 12px -18px -18px;
    padding: 12px 18px calc(12px + env(safe-area-inset-bottom));
    border-top: 1px solid var(--border);
    background: rgba(17, 24, 28, .96);
    backdrop-filter: blur(16px);
  }
  .glb-preview { height: 440px; min-height: 300px; }
  .control-grid { grid-template-columns: minmax(0, 1fr); }
  .section-head { align-items: start; flex-direction: column; }
  .hide-mobile { display: none !important; }
}

@media (max-width: 430px) {
  body { font-size: 16px; }
  .brand-text { max-width: 172px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  h1 { font-size: 31px; }
  .wrap { padding-inline: 12px; }
  .card, .hero { padding-inline: 16px; }
  .project-thumb { width: calc(100% + 32px); margin-inline: -16px; }
  .hero-actions, .actions { align-items: stretch; flex-direction: column; }
  .hero-actions > *, .actions > *, button, .button { width: 100%; }
  .timeline .stage { flex-basis: 118px; }
  .glb-preview { height: 360px; }
  .decision-more .actions { grid-template-columns: minmax(0, 1fr); }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; }
}
"""
