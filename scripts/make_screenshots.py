"""Load the static site in headless Chromium at desktop and phone sizes, check for console errors,
and save screenshots. Usage: python scripts/make_screenshots.py [outdir] [--live-url http://127.0.0.1:8103]
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8103


def serve(directory: Path):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args, **kwargs) -> None:
            pass

    handler = functools.partial(Quiet, directory=str(directory))
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    srv = serve(ROOT / "docs")
    base = f"http://127.0.0.1:{PORT}/"
    errors: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for name, ctx_args in {
                "desktop": dict(viewport={"width": 1440, "height": 900}),
                "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                              is_mobile=True, has_touch=True),
            }.items():
                ctx = browser.new_context(**ctx_args)
                page = ctx.new_page()
                page.on("console", lambda m, n=name: errors.append(f"{n}: {m.text}")
                        if m.type in ("error", "warning") and not m.location.get("url", "").endswith("/health") else None)
                page.on("pageerror", lambda e, n=name: errors.append(f"{n} pageerror: {e}"))
                page.goto(base, wait_until="networkidle")
                page.wait_for_selector("#graph .node")
                page.screenshot(path=str(out / f"{name}-start.png"))
                page.evaluate("document.querySelector('#lab').scrollIntoView()")
                max_step = int(page.get_attribute("#scrub", "max"))
                page.evaluate(f"document.querySelector('#scrub').value={max_step};"
                              "document.querySelector('#scrub').dispatchEvent(new Event('input'))")
                page.wait_for_timeout(1200)
                page.evaluate("document.querySelector('#lab').scrollIntoView()")
                page.screenshot(path=str(out / f"{name}-graph.png"))
                nodes = page.query_selector_all("#graph .node.keep")
                target = nodes[-1] if nodes else page.query_selector("#graph .node")
                target.dispatch_event("click") if name == "desktop" else target.tap()
                page.wait_for_timeout(900)
                page.screenshot(path=str(out / f"{name}-detail.png"))
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(300)
                overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
                if overflow > 0:
                    errors.append(f"{name}: page scrolls sideways by {overflow}px")
                page.screenshot(path=str(out / f"{name}-full.png"), full_page=True)
                ctx.close()
            browser.close()
    finally:
        srv.shutdown()
    for e in errors:
        print("ISSUE", e)
    print("screenshots in", out, "| issues:", len(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
