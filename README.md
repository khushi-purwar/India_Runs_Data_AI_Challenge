# Redrob Candidate Ranker — Senior AI Engineer (Founding Team)

A hybrid, fully-offline system that ranks the top 100 candidates from a 100,000-profile
pool for the Redrob AI **"Senior AI Engineer — Founding Team"** job description.

It ranks the way the JD asks a recruiter to: by **understanding who fits**, not by counting
AI keywords. It rejects keyword-stuffer traps, surfaces plain-language "Tier 5" engineers,
down-weights unavailable candidates, and hard-filters impossible **honeypot** profiles.

```
python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```

Runtime: **~55 seconds for 100K candidates** on a CPU-only 16 GB machine. No GPU, no network.

---

## Why this design

The JD is deliberately written to defeat naive matching. Three failure modes are baked into
the dataset, and the architecture targets each directly:

| Trap | What it looks like | How we beat it |
|------|--------------------|----------------|
| **Keyword stuffers** | "Marketing Manager" / "Accountant" with every AI skill listed | Title fit is a decisive, weighted component; non-tech current titles are capped. Skills are trust-weighted (endorsements + duration + assessment), so a stuffed list counts for little. |
| **Tier-5 candidates** | Built recsys/search at a product company but never says "RAG" or "Pinecone" | Domain score reads **career prose** (3× the weight of the skills array) and blends a **TF-IDF semantic match** against a plain-language JD query, surfacing real work without the vocabulary. |
| **Honeypots (~80)** | 8 yrs at a 3-yr-old company; "expert" in 10 skills with 0 months used | A dedicated [`honeypot.py`](ranker/honeypot.py) detector hard-zeros internally-impossible profiles. Result: **0% honeypot rate in the top 100**. |

The JD's explicit disqualifiers (consulting-only careers, CV/speech-only specialists, title
chasers, academia-only) are applied as multiplicative penalties, and its behavioral guidance
(an unreachable candidate "is not actually available") becomes an availability multiplier.

---

## Architecture

```
candidates.jsonl ──▶ text.py ──▶ semantic.py (TF-IDF cosine vs JD query)
                       │                       │
                       ▼                       ▼
                  scoring.py ◀── jd.py (structured JD signal vocabulary)
                       │
        ┌──────────────┼───────────────────────────────┐
        ▼              ▼                                 ▼
  7 component      behavioral                    disqualifier
   scores          multiplier                     multiplier
 (weighted sum)   (availability)                 (JD anti-signals)
        └──────────────┼───────────────────────────────┘
                       ▼
              honeypot.py hard-zero ──▶ rank ──▶ reasoning.py ──▶ submission.csv
```

### Scoring components (weighted sum → base fit)

| Component | Weight | Signal |
|-----------|:------:|--------|
| Domain evidence | 0.30 | Retrieval/ranking/recsys/NLP in career prose + TF-IDF semantic match |
| Title fit | 0.22 | Engineering/ML title vs non-tech (the anti-keyword-stuffer signal) |
| Trusted skills | 0.15 | Relevant skills weighted by endorsements, duration, assessment score |
| Experience | 0.10 | Soft band around the JD's 5–9 years |
| Product vs services | 0.10 | Product-company experience vs consulting-only |
| Location | 0.08 | Pune/Noida/metro India or willing to relocate |
| Education | 0.05 | Institution tier + relevant field |

### Multipliers (applied to base fit)

- **Behavioral availability** (`~0.45–1.08`): last-active recency, recruiter response rate,
  open-to-work, interview completion, profile completeness, verification.
- **Disqualifiers** (`≤1.0`): consulting-only career, CV/speech-only without NLP/IR, title
  chasing (multiple <18-month stints), academia-only with no production evidence.
- **Honeypot guard**: impossible profiles → score `0`.

`final = base_fit × behavioral × disqualifier × honeypot_keep`

---

## Repository layout

```
rank.py                     CLI entry point (the single reproduce command)
app.py                      Streamlit sandbox (Section 10.5 requirement)
requirements.txt
submission_metadata.yaml
ranker/
  jd.py                     Structured JD interpretation + component weights
  text.py                   Candidate text builders (prose vs skills)
  semantic.py               TF-IDF (default) / sentence-transformers (optional)
  scoring.py                Component scorers, multipliers, combine
  honeypot.py               Impossible-profile detection
  reasoning.py              Fact-grounded, varied reasoning strings
  io_utils.py               JSONL/gz loading, CSV writing
```

---

## Reproducing the submission

1. Install dependencies (Python 3.11):

   ```bash
   pip install -r requirements.txt
   ```

2. Place the candidate pool at `India_runs_data_and_ai_challenge/candidates.jsonl`
   (or pass any path; `.jsonl.gz` is also accepted).

3. Run:

   ```bash
   python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
   ```

4. Validate the format:

   ```bash
   python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
   # -> Submission is valid.
   ```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--candidates` | (required) | Path to `candidates.jsonl` / `.jsonl.gz` |
| `--out` | `submission.csv` | Output CSV path |
| `--top` | `100` | Number of candidates to output |
| `--semantic` | `tfidf` | `tfidf` (offline default) or `st` (sentence-transformers; needs a locally cached model, no network at ranking time) |

---

## Compute compliance

| Constraint | Limit | This system |
|------------|-------|-------------|
| Runtime | ≤ 5 min | ~55 s for 100K |
| Memory | ≤ 16 GB | streaming load + sparse TF-IDF |
| Compute | CPU only | yes |
| Network | off | no calls at ranking time |

---

## Sandbox

`app.py` is a Streamlit app that accepts a small candidate sample and runs the full ranker
end-to-end on CPU. Deploy it free on Streamlit Cloud or HuggingFace Spaces, or run locally:

```bash
streamlit run app.py
```

---

## Design notes / honest limitations

- TF-IDF is the default semantic engine because it is deterministic, self-contained, and fast
  enough to stay well inside the budget. The dense `--semantic st` path is available when a
  model is cached locally and can improve recall of plain-language matches.
- The JD interpretation in `jd.py` is hand-authored. That is intentional: the highest-leverage
  "understanding" of this role is human judgment about what its words *mean*, encoded as
  signals the machine can apply consistently across 100K profiles.
- Honeypot detection is heuristic (internal inconsistencies), not a hardcoded ID list, so it
  generalizes to unseen impossible profiles.
