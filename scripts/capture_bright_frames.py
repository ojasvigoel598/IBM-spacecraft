"""Capture BRIGHT frames from the live MissionMind dashboard for the demo video.

Temporarily injects a CSS override into the Streamlit page to lighten the
background from #05070f to #0d1117 (GitHub-dark) for much better video
visibility.  The frames are screenshots of the REAL running app — no fake
dashboards.

Run:  .venv/Scripts/python.exe scripts/capture_bright_frames.py [out_dir]
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "demo", "bright_frames"))

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

# JavaScript to inject a <style> element that brightens the Streamlit dark theme
INJECT_JS = r"""
(function() {
  if (document.getElementById('__demo_bright')) return;
  var s = document.createElement('style');
  s.id = '__demo_bright';
  s.textContent = `
    :root {
      --bg: #0d1117 !important;
      --secondary-bg: #161b22 !important;
      --text: #f0f6fc !important;
    }
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"], .stTabs,
    [data-testid="stVerticalBlockBorderWrapper"] {
      background-color: #0d1117 !important;
      color: #f0f6fc !important;
    }
    .stApp > header { background-color: #0d1117 !important; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; }
    h1, h2, h3, h4, h5, h6, p, span, div, label {
      color: #f0f6fc !important;
    }
    .stMetric [data-testid="stMetricValue"] { color: #58a6ff !important; }
    .stMetric [data-testid="stMetricLabel"] { color: #8b949e !important; }
    [data-testid="stDataFrame"] { border: 1px solid #30363d !important; }
  `;
  document.head.appendChild(s);
})();
"""


def find_chrome():
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def shot(page, name, out_dir):
    path = os.path.join(out_dir, f"{name}.png")
    page.screenshot(path=path, full_page=False)
    print(f"  captured {name} -> {os.path.relpath(path, ROOT)}")
    return path


def click_text(page, text, timeout_ms=20000):
    loc = page.get_by_text(text, exact=False).first
    loc.wait_for(state="visible", timeout=timeout_ms)
    loc.click()
    time.sleep(2)


def brighten(page):
    """Inject CSS to brighten the dark Streamlit theme."""
    page.evaluate(INJECT_JS)
    page.wait_for_timeout(800)


def main():
    chrome = find_chrome()
    if not chrome:
        print("FATAL: system Chrome not found")
        return 1

    os.makedirs(OUT, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 960}, device_scale_factor=1)
        page = ctx.new_page()

        print("=== Loading Streamlit dashboard ===")
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        page.wait_for_timeout(18000)  # let Streamlit fully render

        # Inject brightness CSS
        brighten(page)

        # Wait for key UI elements
        try:
            page.wait_for_selector("text=System Status", timeout=30000)
        except Exception:
            print("  Warning: System Status not found, waiting more...")
            page.wait_for_timeout(10000)

        print("\n=== Capturing frames ===")

        # 1. Normal operation
        brighten(page)
        shot(page, "01_normal", OUT)

        # 2. Solar fault - click scenario
        click_text(page, "Solar Array Degradation")
        page.wait_for_timeout(6000)
        brighten(page)
        shot(page, "02_solar_fault", OUT)

        # 3. Fault detection moment
        click_text(page, "Fault onset")
        page.wait_for_timeout(5000)
        brighten(page)
        shot(page, "03_detection", OUT)

        # 4. Deep fault - advance time
        for _ in range(2):
            click_text(page, "+5 min")
            page.wait_for_timeout(1500)
        page.wait_for_timeout(3000)
        brighten(page)
        shot(page, "04_deep_fault", OUT)

        # 5. RAG evidence
        click_text(page, "RAG & Evidence")
        page.wait_for_timeout(6000)
        brighten(page)
        shot(page, "05_rag_evidence", OUT)

        # 6. Granite reasoning
        click_text(page, "Granite Reasoning")
        page.wait_for_timeout(6000)
        brighten(page)
        shot(page, "06_granite", OUT)

        # 7. 3D spacecraft - scroll down
        click_text(page, "Telemetry")
        page.wait_for_timeout(3000)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(5000)
        brighten(page)
        shot(page, "07_threejs", OUT)

        print(f"\nDONE — frames captured in {OUT}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
