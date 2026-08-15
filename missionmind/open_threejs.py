"""
Short runner to open three_spacecraft_standalone.html with real physics simulation
- Starts HTTP server (so Three.js importmap works, not file:// CORS)
- Opens browser to http://localhost:8000/three_spacecraft_standalone.html
- Physics is real: JS loop mirrors Python simulator (P_solar, net, dSOC, Q_out, dT)

Run: python -m missionmind.open_threejs
or: python missionmind/open_threejs.py
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import threading
import time

# P3-006 FIX: printed math symbols (ε, σ) crash the Windows cp1252 console — force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PORT = 8000
DIR = os.path.join(os.path.dirname(__file__), "viz", "components")
FILE = "three_spacecraft_standalone.html"

def start_server():
    os.chdir(DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving {DIR} at http://localhost:{PORT}/{FILE}")
        print("Physics is REAL simulation in JS — not fake animation:")
        print("  solar = 520*degradation(t), net=solar-400, dSOC=net/3600/100, soc=clamp(soc+dSOC)")
        print("  qOut=εσA(T^4-3^4), dT=(60-qOut)/2000, T+=dT")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start server in thread
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    time.sleep(1)
    url = f"http://localhost:{PORT}/{FILE}"
    print(f"Opening {url}")
    webbrowser.open(url)
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped")
