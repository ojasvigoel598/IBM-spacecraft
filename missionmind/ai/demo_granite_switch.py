"""
One-file demo to show IBM Granite switch (real vs mock)
Set env vars for real watsonx, else mock with RAG citations still works.

Run: python -m missionmind.ai.demo_granite_switch

"""

import os
# Uncomment and fill to test real watsonx:
# os.environ["WATSONX_APIKEY"] = "YOUR_KEY"
# os.environ["WATSONX_PROJECT_ID"] = "YOUR_PROJECT_ID"

from missionmind.ai.prompts import example_input_json
from missionmind.ai.granite_client import generate_explanation, WATSONX_AVAILABLE

print(f"SDK available: {WATSONX_AVAILABLE}")
print(f"API key present: {bool(os.getenv('WATSONX_APIKEY'))}")
print(f"Project present: {bool(os.getenv('WATSONX_PROJECT_ID'))}")

inp = example_input_json()
print("\nInput:", inp)
print("\nGenerating with RAG...")
out = generate_explanation(inp, use_rag=True, top_k=3)
import json
print(json.dumps(out, indent=2))
print("\nSchema check: risk" , out.get("risk"), "evidence", out.get("evidence_used"))
