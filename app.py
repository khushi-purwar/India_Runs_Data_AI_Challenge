"""Streamlit sandbox for the Redrob ranker.

Satisfies the hackathon sandbox requirement (Section 10.5): a hosted environment
that accepts a small candidate sample (<=100 candidates) and runs the ranking
system end-to-end on CPU within the compute budget.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Cloud or HuggingFace Spaces by pointing it at this repo.
"""
from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from ranker.reasoning import build_reasoning
from ranker.scoring import score_candidate
from ranker.semantic import build_semantic
from ranker.text import full_text

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Candidate Ranker — Senior AI Engineer (Founding Team)")
st.caption(
    "Hybrid semantic + rule-based ranker. Upload a small JSON/JSONL candidate "
    "sample (<=100 records) and the system ranks them end-to-end on CPU."
)


def load_records(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8")
    text_stripped = text.lstrip()
    if text_stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


uploaded = st.file_uploader(
    "Candidate sample (.json array or .jsonl)", type=["json", "jsonl"]
)
top_n = st.slider("How many to rank", 5, 100, 25)

if uploaded is not None:
    candidates = load_records(uploaded.read())
    candidates = candidates[:100]
    st.write(f"Loaded **{len(candidates)}** candidates.")

    with st.spinner("Building semantic index and scoring..."):
        docs = [full_text(c) for c in candidates]
        sem = build_semantic("tfidf")
        sem.fit_transform(docs)
        sims = sem.similarities()

        scored = []
        for c, sim in zip(candidates, sims):
            scored.append((c, score_candidate(c, float(sim))))
        for _, r in scored:
            r["round_score"] = round(r["score"], 4)
        scored.sort(key=lambda cr: (-cr[1]["round_score"], cr[0]["candidate_id"]))

    rows = []
    for rank, (c, r) in enumerate(scored[:top_n], start=1):
        rows.append({
            "rank": rank,
            "candidate_id": c["candidate_id"],
            "score": r["round_score"],
            "honeypot": "YES" if r["honeypot_reasons"] else "",
            "reasoning": build_reasoning(c, r, rank),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    df[["candidate_id", "rank", "score", "reasoning"]].to_csv(csv_buf, index=False)
    st.download_button("Download ranked CSV", csv_buf.getvalue(),
                       file_name="ranked_sample.csv", mime="text/csv")
else:
    st.info("Upload sample_candidates.json from the hackathon bundle to try it.")
