#!/bin/bash
echo "Stopping MissionMind demo servers..."
lsof -ti:8501 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
pkill -f "streamlit run" || true
pkill -f "http.server 8000" || true
echo "Stopped"
