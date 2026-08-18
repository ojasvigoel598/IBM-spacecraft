"""
MissionMind - RAG Retrieval
Simple TF-IDF based retriever over markdown docs in knowledge_base.

Produces evidence for Granite: retrieves spacecraft subsystem documentation,
troubleshooting procedures, mission rules.

Production design: no external API needed, works offline; can be upgraded to embeddings.

P3-003 FIX: Standardize DOC IDs to DOC-XXX-### format, replace underscores, uppercase
"""

import os
import glob
import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(__file__), 'knowledge_base')

class RAGRetriever:
    def __init__(self, kb_dir: str = KB_DIR):
        self.kb_dir = kb_dir
        self.documents = []  # list of dict {id, title, content, path}
        self.vectorizer = None
        self.doc_vectors = None
        self._load_docs()
        self._build_index()

    def _load_docs(self):
        pattern = os.path.join(self.kb_dir, "*.md")
        files = sorted(glob.glob(pattern))
        docs = []
        seen_ids = {}
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            lines = content.strip().split('\n')
            root_title = lines[0].replace('#', '').strip() if lines else os.path.basename(fp)
            file_base = os.path.splitext(os.path.basename(fp))[0].upper()
            # Split into level-2 (##) sections; within each, split level-3
            # (###) subsections so every ID-carrying block ([DOC-...]) becomes
            # its own retrievable chunk. A citation must point at the text
            # that actually carries the ID, so the ID is extracted from the
            # chunk's OWN text - never by positional lookup over the whole
            # document (which mislabels every section after the header).
            sections = re.split(r'\n##\s+', content)
            idx = 0
            for sec in sections:
                if not sec.strip():
                    continue
                for sub in re.split(r'\n###\s+', sec):
                    if not sub.strip():
                        continue
                    idx += 1
                    chunk_text = sub.strip()
                    # Skip pure section skeletons (a heading with no body)
                    # - they carry no evidence and only add retrieval noise.
                    if len(chunk_text) < 40:
                        continue
                    ids = re.findall(r'\[([A-Z0-9\-]+)\]', chunk_text)
                    sec_id = ids[0] if ids else f"{file_base}-{idx}"
                    sec_id = sec_id.replace('_', '-').upper().strip()
                    if not sec_id.startswith("DOC"):
                        sec_id = f"DOC-{sec_id}"
                    sec_id = re.sub(r'[^A-Z0-9\-]', '-', sec_id)
                    sec_id = re.sub(r'-+', '-', sec_id)
                    # Duplicate IDs break citation provenance (a reader could
                    # not tell which chunk a [DOC-...] refers to), so make
                    # duplicates unique deterministically and say so.
                    if sec_id in seen_ids:
                        seen_ids[sec_id] += 1
                        sec_id = f"{sec_id}-{seen_ids[sec_id]}"
                        print(f"[RAG] warning: duplicate chunk id in {fp}, "
                              f"renamed to {sec_id}")
                    else:
                        seen_ids[sec_id] = 1
                    heading = chunk_text.split('\n', 1)[0][:80]
                    docs.append({
                        "id": sec_id,
                        "title": f"{root_title} - {heading}",
                        "content": chunk_text[:2000],
                        "path": fp,
                        # Source provenance: which file this chunk came from,
                        # surfaced in prompts and the alert API so a citation
                        # is traceable to a real file (no hallucinated paths).
                        "source": os.path.basename(fp),
                        "section": heading,
                    })
        self.documents = docs
        print(f"[RAG] Loaded {len(docs)} chunks from {len(files)} files")

    def _build_index(self):
        if not self.documents:
            return
        corpus = [d["content"] for d in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top_k docs relevant to query.
        Query is constructed from anomaly report.
        Returns list sorted by score desc.
        """
        if not self.documents or self.vectorizer is None:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_vectors).flatten()
        # Get top indices
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for i in top_indices:
            if scores[i] < 0.05:  # threshold
                continue
            doc = self.documents[i].copy()
            doc["score"] = float(scores[i])
            results.append(doc)
        return results

    def query_from_anomaly(self, anomaly_input: dict, top_k: int = 3):
        """
        Build query string from anomaly dict, retrieve.
        """
        subsystem = anomaly_input.get("subsystem", "")
        flag = anomaly_input.get("physics_flag", "")
        # Build descriptive query
        q_parts = [
            subsystem,
            flag,
            str(anomaly_input.get("current_values","")),
            str(anomaly_input.get("probable_cause","")) if "probable_cause" in anomaly_input else "",
        ]
        # Add logic: if power, query solar, battery etc.
        if subsystem == "power" or (flag and "solar" in flag):
            q_parts.append("solar array degradation battery voltage SOC troubleshooting power load shedding")
        if subsystem == "thermal" or (flag and "radiator" in flag):
            q_parts.append("radiator degradation thermal temperature heat rejection epsilon area troubleshooting")
        q_parts.append("mission rules risk recommended action")
        query = " ".join([p for p in q_parts if p])
        hits = self.retrieve(query, top_k=top_k)
        try:
            from missionmind.trace import record
            record("ai.rag", "query_from_anomaly",
                   note=f"{subsystem or 'unknown'} query -> {len(hits)} docs",
                   value=round(hits[0]["score"], 3) if hits else None)
        except Exception:  # noqa: BLE001
            pass
        return hits

# Singleton for app
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever

if __name__ == "__main__":
    r = RAGRetriever()
    test_input = {
        "subsystem": "power",
        "physics_flag": "solar_degradation",
        "current_values": {"solar_power_w": 248, "battery_voltage_v": 24.6}
    }
    docs = r.query_from_anomaly(test_input, top_k=3)
    for d in docs:
        print(f"[{d['id']}] score={d['score']:.3f} title={d['title']}")
        print(d['content'][:300])
        print("---")
