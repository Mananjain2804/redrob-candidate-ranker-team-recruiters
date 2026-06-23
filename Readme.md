# 🚀 Redrob Hackathon – Intelligent Candidate Discovery & Ranking Challenge

This repository contains our submission for the Redrob hackathon. Our ranking system selects the top 100 candidates for the given job description using a **tiered scoring architecture** that combines structured feature extraction with multiplicative dealbreaker penalties — designed to closely mirror human recruiter judgment.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point – loads candidates, runs ranking, and outputs the final CSV. |
| `ranker.py` | **Layer 2** – Weighted scoring formula and multiplicative penalty engine. |
| `feature_engineering.py` | **Layer 1** – Extracts 12 positive features and 11 dealbreaker signals per candidate. |
| `reasoning.py` | Generates 1–2 sentence factual reasoning for each ranked candidate. |
| `config.py` | All pattern sets, scoring weights, penalty factors, and thresholds. |
| `compute_metrics.py` | Ground truth correlation analysis (Pearson vs. human labels). |
| `validate_submission.py` | Format validator (provided in the challenge bundle). |
| `hybrid_ranker.py` | Hybrid ranking utilities. |
| `data/candidates.zip` | **Compressed candidate dataset** (~54 MB, tracked in Git). |
| `data/sample_candidates.json` | Small sample (50 candidates) for quick testing. |
| `data/candidate_schema.json` | JSON schema for a single candidate record. |
| `data/sample_submission.csv` | Reference submission format. |
| `data/submission_metadata_template.yaml` | Metadata template required for submission. |
| `Team_recruiters_submission.csv` | Our final generated submission. |

> **Note on large files**: The raw `candidates.jsonl` (~464 MB) and `data.zip` (~108 MB) are excluded via `.gitignore` because they exceed GitHub's 100 MB file-size limit. The pipeline reads directly from `data/candidates.zip` (~54 MB), which is fully tracked in this repo.

---

## 🚀 Setup and Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Data is ready to use** – `data/candidates.zip` is already included in the repo.  
   No extra download step is needed. The pipeline extracts it on-the-fly.

---

## 🧠 Approach Overview

Our system uses a **two-layer scoring architecture** inspired by a diagnostic analysis of human recruiter rankings. The core insight: a human ranker weights *implicit signals* (cultural fit, coding recency, domain misalignment, non-competes) as heavily as explicit skill matches — our old ranker missed all of these.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Feature Extraction  (feature_engineering.py)  │
│  ────────────────────────────────────────────────────── │
│  12 Positive Feature Scorers  →  each returns [0, 1]    │
│  11 Dealbreaker Detectors     →  each returns bool      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Weighted Scoring  (ranker.py)                 │
│  ────────────────────────────────────────────────────── │
│  Base Score  = Σ (weight_i × feature_i)                 │
│  Penalty     = Π (penalty_factor_j)                     │
│  Final Score = Base Score × Penalty Multiplier           │
│  Tie-break   = candidate_id ascending                   │
└─────────────────────────────────────────────────────────┘
```

### Layer 1: Positive Feature Scorers

These 12 features capture what makes a candidate *good* for the Senior AI Engineer (Ranking/Retrieval) role:

| Feature | Weight | What It Captures |
|---------|--------|------------------|
| `retrieval_score` | 0.18 | Core retrieval/ranking/embedding keyword match depth |
| `production_fit` | 0.16 | Vector DB + hybrid search + eval frameworks + shipping composite |
| `evaluation_score` | 0.12 | NDCG, MRR, precision@k, A/B testing evidence |
| `pre_llm_score` | 0.10 | Pre-2022 search/ranking experience + classic ML tools (xgboost, lightgbm, elasticsearch, learning-to-rank) |
| `ai_yoe_score` | 0.08 | Total years in AI/ML roles |
| `ranking_yoe_score` | 0.08 | Years specifically in ranking/search/retrieval job titles |
| `experience_fit` | 0.06 | YOE sweet-spot (5–9 years ideal) |
| `recent_coder_score` | 0.06 | Hands-on coding evidence within 18 months (GitHub activity, technical titles) |
| `location_score` | 0.05 | Pune/Noida preferred → metro → relocatable → other |
| `notice_score` | 0.04 | ≤30 days best, >90 days worst |
| `response_score` | 0.04 | Recruiter response rate + activity recency + response time |
| `shipping_score` | 0.03 | Track record of deploying/shipping production systems |

### Layer 1: Dealbreaker Detectors

These 11 penalties capture what makes a candidate *risky* — the "hidden signals" that human recruiters weigh heavily but simple keyword matchers miss:

| Dealbreaker | Penalty Factor | Detection Logic |
|-------------|---------------|-----------------|
| **Honeypot** | 0.00 (instant zero) | Fictional companies (Dunder Mifflin, Stark Industries), YOE vs career-history mismatch ≥1 year, expert skills with 0 months usage, keyword-stuffed summary without technical titles |
| **Non-Compete** | 0.15 | Non-compete / restrictive covenant / garden leave / exclusivity clause in text |
| **CV/Speech Domain** | 0.30 | ≥3 computer vision/speech/robotics terms AND ≤1 ranking/retrieval term |
| **Manager-Only** | 0.40 | Director/VP/Manager title + no recent hands-on coding evidence |
| **Consultancy-Only** | 0.50 | Entire career at service companies (TCS, Infosys, Wipro, Accenture, etc.) |
| **Culture Misfit** | 0.50 | "stable environment", "predictable schedule", "mature codebase" — misaligned with high-velocity shipping culture |
| **Job Hopper** | 0.55 | Average tenure < 18 months across career |
| **Flight Risk** | 0.55 | Offer acceptance rate < 20% |
| **Framework Enthusiast** | 0.60 | LangChain/LlamaIndex present but zero evaluation framework evidence |
| **Ghost** | 0.35 | Recruiter response rate < 25% + inactive > 60 days |
| **Research-Only** | 0.60 | Research-heavy background (publications, academic papers) with low shipping evidence (shipping score ≤ 0.20) |

> **Key Design Decision**: Penalties are *multiplicative*, not additive. A candidate with two dealbreakers (e.g., consultancy-only × job-hopper = 0.50 × 0.55 = 0.275) gets hit much harder than one with a single dealbreaker. This mirrors how human recruiters "stack" concerns.

### Layer 2: Final Score Computation

```python
final_score = base_score × penalty_multiplier
# where:
#   base_score = Σ (POSITIVE_WEIGHTS[feat] × feature_score[feat])
#   penalty_multiplier = Π (penalty_factor for each triggered dealbreaker)
```

All weights and penalty factors are defined as constants in `config.py` for easy tuning.

---

## 🔍 Reasoning Generation

For each candidate in the top 100, we produce a unique reasoning string that:

1. **Zero Hallucination**: Only references skills, companies, tools, and locations verified to exist in the candidate's JSON profile.
2. **Specific Facts**: Cites exact YoE, AI/ML specialization duration, named vector DBs (Pinecone, Faiss, etc.), evaluation frameworks (NDCG, MRR), current title and company.
3. **Honest Concerns**: Surfaces all applicable dealbreaker flags — non-compete risks, CV domain mismatch, job-hopping patterns, ghost status, flight risk, culture misfit, etc.
4. **Honeypot Flagging**: Explicitly explains *why* a profile is suspicious (fictional company name, YoE mismatch, expert skills with 0 duration).
5. **Layout Variation**: Dynamically selects between 4 paragraph layouts with randomized phrasing to prevent repetitive structures.
6. **Rank-Tone Alignment**: Strong/decent/weak tone tiers ensure top-ranked candidates get endorsements while lower-ranked ones get appropriately critical assessments.

---

## ⚙️ How to Run

### Generate the Submission CSV

Using the default path (candidates.zip, already in repo):
```bash
python main.py --out Team_recruiters_submission.csv --top_n 100
```

Or specify the data file explicitly:
```bash
# From the compressed zip (recommended – already in Git)
python main.py --candidates data/candidates.zip --out Team_recruiters_submission.csv --top_n 100

# From a raw JSONL (if you have it locally)
python main.py --candidates data/candidates.jsonl --out Team_recruiters_submission.csv --top_n 100

# Quick test on sample data (50 candidates)
python main.py --candidates data/sample_candidates.json --out test_output.csv --top_n 10
```

Supported formats: `.json`, `.jsonl`, `.jsonl.gz`, `.zip` (containing a `.jsonl`).

### Validate the CSV
```bash
python validate_submission.py Team_recruiters_submission.csv
```
This checks header, column order, row count, rank uniqueness, score monotonicity, and candidate ID format.

---

## ⏱️ Performance & Compute Compliance

| Constraint | Limit | Actual |
|------------|-------|--------|
| **Runtime** | < 5 minutes (CPU) | **51.3 seconds** |
| **Memory** | < 16 GB | **< 2 GB** |
| **Network** | None (offline) | **No external API calls** |
| **GPU** | Not used | **CPU only** |

### Performance Optimizations

- **Pre-compiled regex patterns**: All keyword sets are compiled once into alternation patterns (`\b(?:term1|term2|...)\b`) and cached in `_PATTERN_CACHE`. This is ~8× faster than per-term `re.search` loops.
- **LRU-cached text normalization**: `normalize_text()` uses `@functools.lru_cache(maxsize=512)` to avoid re-normalizing the same text blocks across multiple feature extractors.
- **Shared text builders**: `_full_text()` and `_career_text()` helper functions avoid redundant string concatenation across feature extractors.
- **Early exit on honeypots**: `is_honeypot()` is checked first in `score_candidate()` — returning 0.0 immediately skips all 12 positive feature computations.

---

## 📊 Results

### Score Distribution (100K candidates → Top 100)

| Metric | Value |
|--------|-------|
| Top score (Rank 1) | **0.9134** |
| 25th percentile | **0.7864** |
| Median (50th) | **0.7256** |
| 75th percentile | **0.6766** |
| 100th score | **0.6619** |
| Score spread | **0.2515** |

### Ground Truth Validation

Spot-check against human-labeled candidates:

| Candidate | Human Label | In Top 100? | Rank | Score |
|-----------|------------|-------------|------|-------|
| CAND_0048558 | **2** (best) | ✅ Yes | 56 | 0.7056 |
| CAND_0096104 | **2** (best) | ✅ Yes | 60 | 0.6914 |
| CAND_0064904 | 1 (good) | ✅ Yes | 79 | 0.6727 |
| CAND_0061257 | 1 (good) | ✅ Yes | 100 | 0.6619 |
| CAND_0007009 | **0** (bad) | ❌ No | — | — |

Label=2 candidates consistently rank above label=1 candidates. The label=0 candidate is correctly excluded from the top 100.

---

## 📝 Reproducing Our Exact Submission

```bash
python main.py --candidates data/candidates.zip --out Team_recruiters_submission.csv --top_n 100
python validate_submission.py Team_recruiters_submission.csv
```

---

## 🤖 AI Tools Declaration

We used AI tools as part of our development workflow:
- **Claude / Gemini** – for architecture discussions, code reviews, diagnostic analysis, and reasoning engine design.
- **GitHub Copilot** – for autocompletion and boilerplate generation.

**No candidate data** was sent to any LLM at any point. The final ranking logic is entirely deterministic and reproducible offline.

---

## 🧪 Dependencies

See `requirements.txt`. Key packages:
- Python ≥ 3.8
- scikit-learn
- numpy

---

## ⚠️ Large File Notice

GitHub enforces a **100 MB per-file limit**. The following files are excluded from this repository via `.gitignore`:

| File | Size | Reason |
|------|------|--------|
| `data/candidates.jsonl` | ~464 MB | Exceeds GitHub limit |
| `data.zip` | ~108 MB | Exceeds GitHub limit |

**What to use instead**: `data/candidates.zip` (~54 MB) is included in the repo and contains the same candidate data. The pipeline handles it natively – no changes needed.

---

## 📬 Contact

For questions about this repository, please reach out to the team via the hackathon portal.
