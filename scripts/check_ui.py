"""Headless-Chromium check of the viewer, in served mode (FastAPI) and static mode (docs/ from a plain
file server), at 1440x900 and at 390x844 with touch. Reports console errors, failed requests,
sideways page scroll, small tap targets and a covered "RECORDED RUN" line, and saves screenshots.

    python scripts/check_ui.py [--shots docs/assets] [--modes served static] [--port 8103]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VIEWPORTS = {
    "desktop": dict(viewport={"width": 1440, "height": 900}),
    "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True),
}
IGNORED_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com")


def start_static(port: int):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k) -> None:
            pass

    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", port), functools.partial(Quiet, directory=str(ROOT / "docs")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return lambda: (srv.shutdown(), srv.server_close())


def start_served(port: int):
    proc = subprocess.Popen([sys.executable, "-m", "promptlab.cli", "serve", "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            break
        except OSError:
            time.sleep(0.5)
    return lambda: (proc.terminate(), proc.wait(timeout=10))


# Screenshots the README embeds: only recorded-run (static) captures, nothing else is written.
KEPT_SHOTS = {("desktop", "graph"), ("desktop", "operators"), ("phone", "detail")}


def snap(page: Page, shots: Path, vname: str, mode: str, name: str) -> None:
    if mode != "static" or (vname, name) not in KEPT_SHOTS:
        return
    path = str(shots / f"{vname}-{name}.png")
    if name == "operators":  # cut at the card boundary, not through the next section
        page.locator("#operators").screenshot(path=path)
    else:
        page.screenshot(path=path)


def tap_targets(page: Page) -> list[str]:
    return page.evaluate(
        """() => {
          const bad = [];
          const sel = 'button, a[href], input, select, summary, [role=button], [role=tab], .node .hit';
          for (const e of document.querySelectorAll(sel)) {
            const r = e.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) continue;
            const cs = getComputedStyle(e);
            if (cs.visibility === 'hidden' || cs.display === 'none') continue;
            if (e.closest('#detail:not(.open)') && innerWidth <= 960) continue;
            if (e.closest('footer')) continue;
            if (r.width < 43.5 || r.height < 43.5) bad.push(`${e.tagName.toLowerCase()}#${e.id || ''}.${(e.getAttribute('class') || '').split(' ')[0]} ${Math.round(r.width)}x${Math.round(r.height)} "${(e.textContent || '').trim().slice(0, 20)}"`);
          }
          return bad;
        }"""
    )


def run_mode(browser, mode: str, base: str, shots: Path | None, issues: list[str]) -> None:
    for vname, vargs in VIEWPORTS.items():
        tag = f"{mode}/{vname}"
        ctx = browser.new_context(**vargs)
        page = ctx.new_page()
        page.on("console", lambda m, t=tag: issues.append(f"{t}: console {m.type}: {m.text}")
                if m.type in ("error", "warning") and not any(h in (m.location.get("url") or "") for h in IGNORED_HOSTS) else None)
        page.on("pageerror", lambda e, t=tag: issues.append(f"{t}: pageerror: {e}"))
        page.on("response", lambda r, t=tag: issues.append(f"{t}: HTTP {r.status} {r.url}")
                if r.status >= 400 and not any(h in r.url for h in IGNORED_HOSTS) else None)
        page.on("requestfailed", lambda r, t=tag: issues.append(f"{t}: request failed {r.url}")
                if not any(h in r.url for h in IGNORED_HOSTS) else None)
        page.goto(base, wait_until="networkidle")
        page.wait_for_selector("#graph .node")

        def overflow(label: str, page: Page = page, tag: str = tag) -> None:
            ov = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            if ov > 0:
                issues.append(f"{tag}: page scrolls sideways by {ov}px ({label})")

        overflow("load")
        if shots:
            snap(page, shots, vname, mode, "start")
        # the sticky header must not cover the run label after jumping to the viewer
        page.evaluate("document.querySelector('#lab').scrollIntoView()")
        page.wait_for_timeout(1200)
        hb = page.evaluate("document.querySelector('header.top').getBoundingClientRect().bottom")
        rt = page.evaluate("document.querySelector('#runMeta').getBoundingClientRect().top")
        if rt < hb:
            issues.append(f"{tag}: header covers the run label (header bottom {hb:.0f}, label top {rt:.0f})")
        if not page.inner_text("#runMeta").startswith("RECORDED RUN"):
            issues.append(f"{tag}: run label missing")
        # wait for the replay to reach the held-out step by scrubbing to the end
        page.wait_for_timeout(2500)
        mid = page.evaluate("+document.querySelector('#scrub').max") // 2
        page.evaluate(f"(()=>{{const s=document.querySelector('#scrub');s.value={mid};s.dispatchEvent(new Event('input'))}})()")
        page.wait_for_timeout(1300)
        page.evaluate("document.querySelector('#lab').scrollIntoView()")
        if shots:
            snap(page, shots, vname, mode, "replay")
        page.evaluate("(()=>{const s=document.querySelector('#scrub');s.value=s.max;s.dispatchEvent(new Event('input'))})()")
        page.wait_for_timeout(1500)
        page.evaluate("document.querySelector('#lab').scrollIntoView()")
        if shots:
            snap(page, shots, vname, mode, "graph")
        ind = page.inner_text("#genInd")
        if "of" not in ind:
            issues.append(f"{tag}: generation indicator missing ({ind!r})")
        scrollable = page.evaluate("(()=>{const b=document.querySelector('#graphScroll');return b.scrollWidth>b.clientWidth+2})()")
        if vname == "phone" and not scrollable:
            issues.append(f"{tag}: graph does not scroll inside its container on a phone")
        if vname == "phone" and not page.evaluate("document.querySelector('#graphWrap').classList.contains('fade-r') || document.querySelector('#graphWrap').classList.contains('fade-l')"):
            issues.append(f"{tag}: no edge fade on the scrolling graph")
        # open a node
        nodes = page.query_selector_all("#graph .node.keep")
        target = nodes[-1] if nodes else page.query_selector("#graph .node")
        target.evaluate("e => e.scrollIntoView({block: 'center', inline: 'center'})")
        page.wait_for_timeout(300)
        box = target.bounding_box()
        if vname == "phone":
            page.touchscreen.tap(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        else:
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_timeout(900)
        if "variant" not in page.inner_text("#detailBody"):
            issues.append(f"{tag}: tapping a node did not open its detail")
        if vname == "phone" and "open" not in (page.get_attribute("#detail", "class") or ""):
            issues.append(f"{tag}: detail is not a bottom sheet on phone")
        if shots:
            snap(page, shots, vname, mode, "detail")
        bad = tap_targets(page)
        if vname == "phone":
            issues.extend(f"{tag}: tap target too small: {b}" for b in bad)
        if page.query_selector("#detail.open"):
            page.click("#btnClose")
            page.wait_for_timeout(600)
        # picker: switch task and seed
        tabs = page.query_selector_all("#runTabs .tab")
        if len(tabs) > 1:
            tabs[1].click()
            page.wait_for_timeout(800)
        seeds = page.query_selector_all("#seedTabs .tab")
        if len(seeds) > 1:
            seeds[1].click()
            page.wait_for_timeout(800)
            if "seed 1" not in page.inner_text("#runMeta"):
                issues.append(f"{tag}: seed picker did not load seed 1")
        page.evaluate("document.querySelector('#operators').scrollIntoView()")
        page.wait_for_timeout(1500)
        if shots:
            snap(page, shots, vname, mode, "operators")
        overflow("after interaction")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1200)
        overflow("bottom")
        if mode == "served" and page.is_hidden("#live"):
            issues.append(f"{tag}: live panel not shown in served mode")
        if mode == "static" and page.is_visible("#live"):
            issues.append(f"{tag}: live panel shown in static mode")
        ctx.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=None, help="directory for screenshots")
    ap.add_argument("--modes", nargs="+", default=["served", "static"])
    ap.add_argument("--port", type=int, default=8103)
    args = ap.parse_args()
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for mode in args.modes:
            stop = start_served(args.port) if mode == "served" else start_static(args.port)
            try:
                run_mode(browser, mode, f"http://127.0.0.1:{args.port}/", shots, issues)
            finally:
                stop()
                time.sleep(1)
        browser.close()
    for i in issues:
        print("ISSUE", i)
    print(f"modes {args.modes} | issues: {len(issues)}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
