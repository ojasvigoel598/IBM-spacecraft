# MissionMind — Technology Stack Review

## Current Stack (Verified Installed + Imported)

| Technology | Version | Used By | Purpose |
|---|---|---|---|
| Python | 3.14 | everything | Runtime |
| scikit-learn | 1.9.0 | ml/train.py, ml/detect.py, ml/advanced_models.py | IsolationForest, LOF, OC-SVM, StandardScaler, PCA, MLP |
| numpy | (via sklearn) | everywhere | Numerical computation |
| pandas | (via sklearn) | ml/train.py, ml/detect.py | DataFrame handling |
| scipy | (via sklearn) | ml/metrics.py, drift detection | Statistical tests |
| torch | 2.13.0+cpu | ml/pinn_*.py, ml/advanced_models.py | PINN autograd backprop |
| xgboost | 3.4.0 | ml/advanced_models.py | XGBOD detector |
| pyod | 3.6.4 | ml/advanced_models.py | XGBOD wrapper |
| shap | 0.52.0 | ml/explainability.py | Feature attribution |
| joblib | (via sklearn) | ml/train.py, ml/detect.py | Model persistence |
| ibm-watsonx-ai | 1.6.1 | ai/granite_client.py | Real Granite/watsonx SDK |
| FastAPI | 0.141.1 | viz/api_server.py | JSON API |
| uvicorn | (via FastAPI) | viz/api_server.py | ASGI server |
| Streamlit | 1.61.1 | viz/app.py | Dashboard |
| plotly | (via Streamlit) | viz/app.py | Charts |
| paho-mqtt | (via requirements) | telemetry/ingest.py | MQTT transport |
| React 19 | (web/package.json) | web/src/ | Mission console |
| Vite 8 | (web/package.json) | web/ | Frontend build |
| Tailwind 4 | (web/package.json) | web/src/ | Styling |

## NOT Installed (Verified Absent)

| Technology | Status | Evidence |
|---|---|---|
| LangChain | NOT INSTALLED | 0 imports, 0 matches in codebase |
| LangGraph | NOT INSTALLED | 0 matches |
| LangFlow | NOT INSTALLED | 0 matches |
| FAISS | NOT INSTALLED | 0 imports |
| Chroma | NOT INSTALLED | 0 imports |
| Qdrant | NOT INSTALLED | 0 imports |
| Milvus | NOT INSTALLED | 0 imports |
| Weaviate | NOT INSTALLED | 0 imports |
| pgvector | NOT INSTALLED | 0 imports |
| Elasticsearch | NOT INSTALLED | 0 imports |
| sentence-transformers | NOT INSTALLED | 0 imports |
| Hugging Face | NOT INSTALLED | 0 imports |
| MLflow | NOT INSTALLED | 0 imports |
| DVC | NOT INSTALLED | 0 imports |
| Docker | NOT INSTALLED | 0 Dockerfiles |

---

## Technology-by-Technology Decision

### IBM watsonx / Granite

**Status:** IMPLEMENTED ✅

**What exists:**
- `ibm-watsonx-ai` SDK 1.6.1 installed
- `granite_client.py`: real `_call_watsonx_granite()` + tagged mock fallback
- Model: `ibm/granite-4-h-small` (current, not deprecated)
- Auth: `WATSONX_APIKEY` + `WATSONX_PROJECT_ID` env vars
- State machine: MOCK / REAL_READY / REAL_FAILED
- Strict mode for smoke testing
- Schema validation on response

**What's NOT tested:** Real credentials (user hasn't created IBM account yet)

**Verdict:** KEEP — correctly implemented, ready for real credentials

---

### IBM Bob

**Status:** USED as primary development tool

**Evidence:** Specific Bob-assisted tasks documented in root README "How IBM Bob Was Used" section — power simulator, thermal model, failure injection, ML pipeline, Granite SDK integration, Streamlit/Three.js UI, auth test suite.

**watsonx.ai integration:** Real `ibm-watsonx-ai` SDK wired to `ibm/granite-4-h-small` with honest mock fallback. Credentials optional; dashboard shows which mode is active.

---

### LangChain

**Status:** NOT INSTALLED

**Question:** Does MissionMind need LangChain?

**Analysis:**
1. What problem does it solve? → Standardized retriever/chain interfaces, prompt management
2. Does MissionMind have this problem? → No. The RAG is 4 markdown files, 31 chunks, TF-IDF. A 200-line custom retriever handles it perfectly.
3. Measurable benefit? → None for this corpus size. LangChain adds ~50MB dependency for zero retrieval improvement.
4. Complexity? → Significant. Chain abstraction, callback handlers, memory management.
5. Reproducibility? → LangChain version pins change frequently, breaking chains.

**Verdict:** DO NOT IMPLEMENT — TF-IDF is simpler, faster, and sufficient for 4 files / 31 chunks

---

### LangFlow

**Status:** NOT INSTALLED

**Analysis:**
1. Visual RAG workflow design? → Not needed. The pipeline is: anomaly → query → retrieve → Granite. No complex orchestration.
2. Improves debugging? → No. The pipeline is transparent in Python.
3. Deployment complexity? → Yes. LangFlow requires a separate server.

**Verdict:** DO NOT IMPLEMENT — adds complexity without measurable benefit

---

### Vector Database (FAISS/Chroma/Qdrant/etc.)

**Status:** NOT INSTALLED

**Analysis:**
1. Dataset size? → 4 files, 31 chunks, ~8KB total. Fits in memory trivially.
2. Update frequency? → Never. KB is static engineering documentation.
3. Metadata filtering? → Already implemented via SYSTEM_KEYWORDS + scope gate.
4. Persistence? → In-memory TF-IDF index built at startup (~1ms).
5. Concurrent users? → Singleton retriever, thread-safe read-only.
6. Deployment? → Zero infrastructure. No database server needed.

**Verdict:** DO NOT IMPLEMENT — a vector database for 31 chunks is architectural overhead with zero measurable benefit

---

### Dense Embeddings (sentence-transformers / Hugging Face)

**Status:** NOT INSTALLED

**Analysis:**
1. Would embeddings improve retrieval? → Possibly for larger KBs. For 31 chunks with TF-IDF, the lexical gap is documented but the recall@k is already 0.944.
2. Cost? → Adds ~500MB model download, GPU optional.
3. Measurable benefit? → Need to benchmark. The current TF-IDF achieves Recall@k=0.944, MRR=0.935.

**Verdict:** OPTIONAL FUTURE WORK — document as a potential upgrade path when KB grows beyond ~100 chunks

---

### Reranker

**Status:** NOT INSTALLED

**Analysis:**
1. Would a reranker help? → For 31 chunks with top_k=3, the TF-IDF ranking is already good. A reranker would add latency for marginal improvement.
2. Cost? → Additional model load, inference time.

**Verdict:** DO NOT IMPLEMENT — unnecessary for current corpus size

---

### BM25

**Status:** NOT INSTALLED (but TF-IDF is functionally similar)

**Analysis:**
1. TF-IDF vs BM25? → For this small corpus, they perform similarly. TF-IDF is already implemented and tested.
2. Would BM25 improve retrieval? → Marginal at best for 31 chunks.

**Verdict:** DO NOT IMPLEMENT — TF-IDF is sufficient

---

### PyTorch

**Status:** INSTALLED (2.13.0+cpu)

**Used by:** PINN models (pinn_raissi.py, pinn_torch.py), advanced_models.py

**Verdict:** KEEP — required for physics-informed NN research

---

### XGBoost

**Status:** INSTALLED (3.4.0)

**Used by:** XGBOD detector in advanced_models.py

**Verdict:** KEEP — required for the model zoo comparison

---

### SHAP

**Status:** INSTALLED (0.52.0)

**Used by:** ml/explainability.py for feature attribution

**Verdict:** KEEP — required for explainability

---

### FastAPI

**Status:** INSTALLED (0.141.1)

**Used by:** viz/api_server.py — the authenticated JSON API

**Verdict:** KEEP — production-grade async API

---

### Streamlit

**Status:** INSTALLED (1.61.1)

**Used by:** viz/app.py — the mission-control dashboard

**Verdict:** KEEP — rapid prototyping dashboard

---

### React/Vite

**Status:** CONFIGURED (web/package.json)

**Used by:** web/src/ — the mission-control console

**Verdict:** KEEP — production-grade frontend

---

### Docker

**Status:** NOT INSTALLED

**Analysis:** No Dockerfile exists. Deployment is via Vercel (serverless).

**Verdict:** OPTIONAL FUTURE WORK — add if containerized deployment is needed

---

### MLflow / DVC

**Status:** NOT INSTALLED

**Analysis:** No experiment tracking. The project uses:
- `dataset.json` for dataset fingerprinting
- `models/*.joblib` for model artifacts
- `models/dataset.json` for reproducibility metadata
- Git for version control

**Verdict:** OPTIONAL FUTURE WORK — add if experiment tracking becomes necessary

---

## Final Technology Stack Decision

### KEEP (Currently Justified)
- scikit-learn (ML ensemble)
- numpy/pandas/scipy (numerical)
- torch (PINN research)
- xgboost/pyod (model zoo)
- shap (explainability)
- ibm-watsonx-ai (Granite integration)
- FastAPI (API)
- Streamlit (dashboard)
- React/Vite (console)
- paho-mqtt (MQTT transport)

### DO NOT IMPLEMENT (No Measurable Benefit)
- LangChain (TF-IDF sufficient for 31 chunks)
- LangFlow (no complex orchestration needed)
- Vector databases (31 chunks fit in memory)
- Dense embeddings (TF-IDF recall already 0.944)
- Reranker (unnecessary for small corpus)
- BM25 (TF-IDF functionally equivalent)
- Docker (Vercel serverless sufficient)
- MLflow/DVC (git + dataset.json sufficient)

### OPTIONAL FUTURE WORK
- Dense embeddings (when KB grows beyond ~100 chunks)
- Docker (if containerized deployment needed)
- MLflow (if experiment tracking needed)

---

## RAG Architecture Assessment

### Current Architecture
```
Anomaly Input → query_from_anomaly() → TF-IDF retrieve() → top_k docs → Granite prompt → JSON response
```

### Measured Performance
| Metric | Score | Notes |
|---|---|---|
| Recall@k | 0.944 | 17/18 questions retrieve expected docs |
| Precision@k | 0.370 | Low because top_k=3 includes some noise |
| MRR | 0.935 | First relevant doc usually ranks #1 |
| nDCG@k | 0.900 | Good ranking quality |
| Retrieval failures | 0 | All questions with expected docs find them |
| No-answer violations | 0 | Negative questions correctly return nothing |

### Context Budget
- Max retrieved docs: 3 (top_k parameter)
- Max chars per chunk: 800 (in prompt construction)
- Max total RAG context: ~2400 chars (3 × 800)
- Max prompt size: bounded by Granite's 500 token output limit

### Is TF-IDF Sufficient?
**YES** — for 4 files / 31 chunks / ~8KB total, TF-IDF is:
- Faster than embeddings (no model download)
- Simpler (no GPU required)
- Deterministic (same query → same results)
- Well-tested (18 golden questions, adversarial suite, 10-run stability)

### When Would Embeddings Help?
- KB grows beyond ~100 chunks
- Semantic similarity matters more than lexical matching
- Cross-language queries needed
- Domain-specific embeddings available

---

## Recommendations

1. **Keep current stack** — it's the smallest stack that provides the strongest measurable capability
2. **Document the TF-IDF decision** — explain why embeddings were intentionally not used
3. **Benchmark before upgrading** — if KB grows, benchmark TF-IDF vs embeddings before switching
4. **No new dependencies** — every addition must have a measurable benefit
