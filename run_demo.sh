#!/bin/bash
# MissionMind — 1-command demo that auto-opens Streamlit + Three.js + shows RAG + Granite
# Usage: chmod +x run_demo.sh && ./run_demo.sh

set -e
echo "🛰️  MissionMind — Production Demo Runner"
echo "=========================================="

# 1. Install
echo "📦 Installing requirements..."
pip install -q -r requirements.txt
pip install -q streamlit plotly scikit-learn pandas numpy joblib

# 2. Ground with NASA data (real-world, not just synthetic)
echo "🌍 Grounding parameters with NASA data (public, no API key needed)..."
python -m missionmind.data.nasa_grounding

# 3. Physics verification (so you understand every equation, not black box)
echo "🔬 Physics hand-check..."
python -m missionmind.simulator.physics_verification

# 4. Run full simulation pipeline
echo "⚡ Generating telemetry CSVs (3 scenarios)..."
python -m missionmind.simulator.run_scenarios

echo "🧪 Testing physics sanity..."
python -m missionmind.tests.test_physics
python -m missionmind.physics_rules.test_rules

echo "🤖 Training ML (IsolationForest ensemble)..."
python -m missionmind.ml.train

echo "🧠 Testing RAG + Granite (mock fallback if no watsonx keys, real if present)..."
python -m missionmind.ai.rag
python -m missionmind.ai.granite_client
python -m missionmind.ai.demo_granite_switch

# 5. Show project structure (so you can manually replicate)
echo "📂 Project structure (explainable, each file checkable):"
find missionmind -type f | sort
echo ""
cat missionmind/docs/physics_maths_check.md | head -n 100
echo ""

# 6. Start Streamlit in background (8501) + Three.js standalone HTTP (8000)
echo "🚀 Starting Streamlit (Mission Control v2 with RAG + Granite panels) on http://localhost:8501"
echo "   Features you asked for:"
echo "   - Real physics change analysis (solar 520→249W, net -150W, dSOC, Q_in vs Q_out)"
echo "   - RAG tab shows retrieved docs with scores + citations explaining WHY failed"
echo "   - Granite tab shows system prompt, user prompt with evidence, JSON with risk/cause/reasoning/action"
echo "   - Scenario comparison across normal/solar/radiator"
echo "   - Three.js PBR model driven by live telemetry (not fake)"

# Kill any existing streamlit/http servers on those ports (portable: try lsof, fuser, pkill)
echo "Cleaning previous servers on 8501/8000..."
if command -v lsof &> /dev/null; then
  lsof -ti:8501 | xargs kill -9 2>/dev/null || true
  lsof -ti:8000 | xargs kill -9 2>/dev/null || true
elif command -v fuser &> /dev/null; then
  fuser -k 8501/tcp 2>/dev/null || true
  fuser -k 8000/tcp 2>/dev/null || true
else
  pkill -f "streamlit run" 2>/dev/null || true
  pkill -f "http.server 8000" 2>/dev/null || true
  echo "lsof/fuser not found, used pkill fallback"
fi

# Start streamlit
nohup streamlit run missionmind/viz/app.py --server.port 8501 --server.headless true > /tmp/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "Streamlit PID $STREAMLIT_PID, log /tmp/streamlit.log"
sleep 3

# Start Three.js standalone server
echo "🚀 Starting Three.js standalone (real physics simulation) on http://localhost:8000/three_spacecraft_standalone.html"
cd missionmind/viz/components
nohup python -m http.server 8000 > /tmp/threejs.log 2>&1 &
THREE_PID=$!
cd -
echo "Three.js PID $THREE_PID, log /tmp/threejs.log"
sleep 2

# 7. Auto-open browsers
echo "🌐 Auto-opening browsers..."
if command -v xdg-open &> /dev/null; then
  xdg-open http://localhost:8501 &
  xdg-open http://localhost:8000/three_spacecraft_standalone.html &
elif command -v open &> /dev/null; then
  open http://localhost:8501 &
  open http://localhost:8000/three_spacecraft_standalone.html &
else
  echo "Please manually open: http://localhost:8501 and http://localhost:8000/three_spacecraft_standalone.html"
fi

echo ""
echo "✅ Both servers running!"
echo "   Streamlit:  http://localhost:8501  (Mission Control v2 — check RAG Evidence tab + Granite tab)"
echo "   Three.js:   http://localhost:8000/three_spacecraft_standalone.html  (production PBR, real physics loop)"
echo ""
echo "📹 To record RAG + Granite panels for demo video (≤3 min):"
echo "   1. In Streamlit, select run_solar_failure.csv, drag time to 1500s"
echo "   2. Watch tab 'Real Physics Change' → shows net -150W, dSOC"
echo "   3. Open tab 'RAG Evidence' → shows [DOC-POWER-002] score 0.285 + content"
echo "   4. Open tab 'Granite Explanation' → shows risk HIGH, cause with numbers, reasoning cites [DOC-...], action per procedure"
echo "   5. Use OBS / QuickTime Screen Record to capture: chart anomaly appears → physics flag → Three.js panels dim → RAG evidence → Granite JSON"
echo "   6. Script: 0:00-0:15 problem, 0:15-0:45 solar failure telemetry change, 0:45-1:15 thermal failure, 1:15-1:45 RAG + Granite with citations, 2:30-3:00 architecture + Bob usage"
echo ""
echo "🧠 Explainable AI components:"
echo "   - physics_rules/rules.py: slope calc polyfit, threshold 364W, fully hand-verifiable"
echo "   - ml/train.py: scaler mean/scale printed, feature deviation z-scores shown in UI"
echo "   - ai/rag.py: TF-IDF cosine, not black box, DOC IDs traceable"
echo "   - ai/granite_client.py: prompt locked to JSON with evidence_used, mock → real watsonx switch via env vars"
echo "   - docs/physics_maths_check.md: hand calcs for every equation"
echo ""
echo "🌍 NASA grounding applied: data/nasa_battery_sample.csv (B0005) + grounded_parameters.json"
echo "   Our 100Wh matches 7S2P ~103.6Wh, 24-28V matches 7S Li-ion 22.4-29.4V"
echo ""
echo "To stop servers: kill $STREAMLIT_PID $THREE_PID  or  lsof -ti:8501,8000 | xargs kill -9"
echo ""
echo "Press Ctrl+C to keep servers running in background, or run: ./stop_demo.sh"

# Keep script alive so user can see logs
echo "Tailing Streamlit log (Ctrl+C to exit, servers stay background)..."
tail -f /tmp/streamlit.log
