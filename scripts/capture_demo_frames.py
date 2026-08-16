"""Capture real frames from the LIVE MissionMind dashboard for the demo video.

Uses playwright (already in .venv) against the system Chrome - no browser
download needed. Drives the actual running Streamlit app (http://127.0.0.1:8501)
with real clicks, then screenshots each key state to out_dir.

States captured:
  01_normal          - t=0, Normal Operation
  02_solar_fault     - Solar Array Degradation at fault onset (t=802s)
  03_solar_deep      - Solar fault deep (t=1500s), CRITICAL + RUL countdown
  04_ml_diagnostics  - ML Diagnostics tab
  05_rag_evidence    - RAG & Evidence tab (citations + scores)
  06_granite         - Granite Reasoning tab (evidence-based JSON)
  07_scenarios       - Scenarios comparison tab
  08_live_ingest     - Live Ingest tab (virtual edge node)
  09_threejs         - PBR Three.js digital-twin section (scrolled into view)
  10_console         - Vite React web console (http://localhost:5173)

Run:  .venv/Scripts/python.exe scripts/capture_demo_frames.py [out_dir]
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "demo", "frames"))

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_chrome():
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def shot(page, name, out_dir, full=True):
    path = os.path.join(out_dir, f"{name}.png")
    page.screenshot(path=path, full_page=full)
    print(f"captured {name} -> {os.path.relpath(path, ROOT)}")
    return path


def click_text(page, text, timeout_ms=20000):
    """Click the first visible element containing exact text."""
    loc = page.get_by_text(text, exact=False).first
    loc.wait_for(state="visible", timeout=timeout_ms)
    loc.click()
    time.sleep(1.5)


def main():
    chrome = find_chrome()
    if not chrome:
        print("FATAL: system Chrome not found")
        return 1
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page = ctx.new_page()

        # ---- 1. Dashboard, normal operation -------------------------------
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        # Streamlit is a websocket SPA - give it time to render and score
        page.wait_for_timeout(15000)
        page.wait_for_selector("text=System Status", timeout=60000)
        page.wait_for_timeout(3000)
        shot(page, "01_normal", OUT)

        # ---- 2. Solar fault onset -----------------------------------------
        click_text(page, "Solar Array Degradation")
        page.wait_for_timeout(6000)
        shot(page, "02_solar_fault", OUT)

        # ---- 3. Deep fault: jump to t=1500 via End then transport ----------
        # Click "Fault onset" jump button (t=802) is fine; then use +5min twice
        click_text(page, "Fault onset")
        page.wait_for_timeout(5000)
        shot(page, "03_solar_fault_onset", OUT)
        for _ in range(2):
            click_text(page, "+5 min")
            page.wait_for_timeout(1500)
        page.wait_for_timeout(4000)
        shot(page, "04_solar_deep", OUT)

        # ---- 4. ML Diagnostics tab ----------------------------------------
        click_text(page, "ML Diagnostics")
        page.wait_for_timeout(5000)
        shot(page, "05_ml_diagnostics", OUT)

        # ---- 5. RAG & Evidence tab ----------------------------------------
        click_text(page, "RAG & Evidence")
        page.wait_for_timeout(6000)
        shot(page, "06_rag_evidence", OUT)

        # ---- 6. Granite Reasoning tab -------------------------------------
        click_text(page, "Granite Reasoning")
        page.wait_for_timeout(6000)
        shot(page, "07_granite", OUT)

        # ---- 7. Scenarios tab ---------------------------------------------
        click_text(page, "Scenarios")
        page.wait_for_timeout(6000)
        shot(page, "08_scenarios", OUT)

        # ---- 8. Live Ingest tab -------------------------------------------
        click_text(page, "Live Ingest")
        page.wait_for_timeout(4000)
        shot(page, "09_live_ingest", OUT)

        # ---- 9. Three.js digital twin section (scroll into view) ----------
        click_text(page, "Telemetry")  # back to Telemetry tab
        page.wait_for_timeout(3000)
        page.get_by_text("Live Spacecraft - PBR Three.js").first.scroll_into_view_if_needed()
        page.wait_for_timeout(6000)  # let the canvas animate
        shot(page, "10_threejs", OUT, full=False)

        # ---- 10. Vite React web console -----------------------------------
        page.goto("http://localhost:5173", wait_until="domcontentloaded")
        page.wait_for_timeout(12000)
        try:
            page.wait_for_selector("body", timeout=30000)
            page.wait_for_timeout(4000)
            shot(page, "11_console", OUT, full=True)
        except Exception as e:  # noqa: BLE001
            print(f"console capture failed: {e}")

        browser.close()
    print(f"DONE - frames in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
