# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

This repository contains our submission for the Redrob hackathon. The ranking
system selects the top 100 candidates from **100,000 profiles** for a
**Senior AI Engineer — Search & Ranking** role using a **five-dimension weighted
scoring model**, a **multi-signal trap/penalty layer**, **hard integrity caps**,
and a **pairwise re-ranking pass** — all in pure Python with no external
dependencies.

---

## Repository Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point — loads candidates, runs ranking, writes CSV. |
| `ranker.py` | Scoring engine, hard caps, and pairwise re-ranking. |
| `feature_engineering.py` | All five dimension scorers + trap/penalty detectors + facts extractor. |
| `reasoning.py` | Factual, template-diversified reasoning generator. |
| `config.py` | All constants: keyword lists, dimension weights, caps, thresholds. |
| `requirements.txt` | Dependency list (none — pure Python 3.8+). |
| `validate_submission.py` | Official format validator (provided in the challenge bundle). |
| `submission_metadata.yaml` | Submission metadata mirroring the upload-portal form. |
| `Redrob_Ranker_Sandbox.ipynb` | Google Colab sandbox notebook (small-sample demo). |
| `Team_recruiters.csv` | Final generated submission (top 100). |
| `.gitignore` | Excludes large data files, caches, and temp outputs. |
| `data1/candidates.zip` | Compressed candidate dataset (~54 MB, 100K candidates) — read natively by the pipeline. |
| `data1/sample_candidates.json` | 50-candidate sample for quick testing / sandbox. |
| `data1/candidate_schema.json` | JSON schema for a single candidate record. |

> **Large-file note:** the raw `data1/candidates.jsonl` (~465 MB) exceeds GitHub's
> 100 MB limit and is excluded via `.gitignore`. The pipeline reads
> `data1/candidates.zip` (~54 MB) natively — no extraction needed.

---

## Setup

No installation required beyond Python 3.8+. There are **no external
dependencies** — the pipeline uses only the standard library.

```bash
git clone <your-repo-url>
cd <repo-folder>
```

Place the dataset under `data1/` (the loader reads `.jsonl`, `.jsonl.gz`, or a
`.zip` containing a `.jsonl`).

---

## How to Run

The pipeline defaults to reading `data1/candidates.zip` and writing
`Team_recruiters.csv`, so the simplest reproduce command is:

```bash
# Reproduce the submission (reads data1/candidates.zip by default)
python main.py

# Validate the output
python validate_submission.py Team_recruiters.csv
```

Equivalent explicit form (same result):

```bash
python main.py --candidates data1/candidates.zip --out Team_recruiters.csv --top_n 100
```

Quick test on the bundled 50-candidate sample:

```bash
python main.py --candidates data1/sample_candidates.json --out test_output.csv --top_n 10
```

Supported input formats: `.json`, `.jsonl`, `.jsonl.gz`, `.zip` (containing a
`.jsonl`). All paths resolve against the `data1/` folder structure.

---

## Sandbox / Demo

A hosted **Google Colab notebook** (`Redrob_Ranker_Sandbox.ipynb`) runs the full
pipeline end-to-end on the official 50-candidate sample and produces a ranked CSV
within the compute budget.

**Live link:** https://colab.research.google.com/drive/1DipjNP7NGY8z50tRuyOfa67DsCMiQ7kQ?usp=sharing

To run it: open the link → upload `sample_candidates.json` via the Files panel →
**Runtime → Run all**. The pipeline source is written out via `%%writefile`
cells, so all code stays fully visible in the notebook.

---

## Approach Overview

Human recruiters weigh *implicit* signals — cultural fit, coding recency,
domain alignment, hireability, data integrity — as heavily as keyword matches.
Our ranker encodes both through a weighted multi-dimension base score, a stack
of multiplicative penalties for dealbreakers, hard caps for non-negotiables,
and a pairwise pass that resolves close calls with explicit recruiter-style
rules.

### Architecture

```
Load candidates  →  Honeypot pre-filter (excluded before scoring)
                 →  Five-dimension weighted base score
                 →  Soft multipliers (env-fit, stability)
                 →  Multiplicative penalties (trap, must-have, eval-absence, flight-risk)
                 →  Hard caps (unreachable, visa, integrity)
                 →  Sort by score desc, candidate_id asc
                 →  Pairwise re-ranking on the top 100
                 →  Reasoning generation
                 →  CSV
```

### Five Scoring Dimensions

| Dimension | Weight | What it captures |
|-----------|--------|------------------|
| JD alignment | 0.30 | Vector DB / hybrid search + retrieval & ranking-eval keyword depth (must-have), LLM fine-tuning + HR-tech (nice-to-have). |
| Technical | 0.30 | Evaluation frameworks (NDCG/MRR) 35%, Python fluency 30%, systems thinking 20%, recent-coding recency 15%. |
| Production shipping | 0.25 | Shipped retrieval/ranking evidence (duration-weighted) + pre-LLM bonus + AI-experience quality (YoE sweet-spot × AI career ratio). |
| Availability | 0.08 | Platform recency, notice period (gentle policy), salary-band fit, location with visa nuance. |
| Behavior / reliability | 0.07 | Response rate, response speed, interview completion, offer acceptance, GitHub, recruiter demand. |

`final = (base + bonuses) × env_mult × stability_mult × trap × must_have × eval_absence × flight_risk`,
then capped.

### Trap / Penalty Layer (multiplicative)

| Signal | Effect | Logic |
|--------|--------|-------|
| YoE over-claim | ×0.15 – ×0.70 | `profile.years_of_experience` vs. career-history sum (planted over-claimers in the data). |
| AI keyword stuffing | ×0.20 – ×0.45 | Many AI buzzwords, little career evidence. |
| Zero-duration expert skills | ×0.25 – ×0.80 | Expert/advanced skills with 0 months logged. |
| Copy-paste descriptions | ×0.25 – ×0.85 | Duration-weighted duplicate job descriptions. |
| Culture-statement contradiction | ×0.15 | Summary states preference for "stable codebase / clear specs" (the JD explicitly warns against this). |
| Non-compete clause | ×0.10 | Restrictive-covenant language in free text. |
| Manager-only | ×0.15 | Senior-management title + no production coding in 18 months. |
| Eval-framework absence | ×0.75 | Strong GitHub but no NDCG/MRR language (a major required skill). |
| Flight risk | ×0.55 | 0% offer acceptance while actively applying. |

### Hard Caps

| Cap | Value | Trigger |
|-----|-------|---------|
| Unreachable | 0.65 | Response rate < 0.20 **and** inactive > 90 days. |
| Visa | 0.62 | Located outside India **and** not willing to relocate. |
| Integrity | 0.50 | Trap multiplier ≤ 0.70 (confirmed data-integrity issue). |

### Honeypot Pre-Filter

Candidates with fictional companies, a ≥ 5-year stated-vs-actual experience gap,
four or more zero-duration expert skills, or keyword-stuffed summaries with no
technical role are excluded **before** scoring. On the 100K dataset this removes
**81,546 profiles (81.5%)**, leaving a rankable pool of **18,454**.

### Pairwise Re-Ranking

The top 100 are passed through an insertion-sort pairwise comparison that only
acts on close calls (score gap ≤ 0.06, max move 4 positions). Six ordered rules
decide each pair: integrity flag → shipped-production gap → JD-alignment gap →
notice-period gap → reachability gap → stability gap. Each swap is recorded and
surfaced in the candidate's reasoning text.

### Reasoning Generation

Each top-100 candidate gets a factual justification built only from verified
profile data (zero hallucination). Templates are diversified by usage count so
adjacent rows never share an opening, and concerns/cautions (YoE gaps,
eval-framework absence, flight risk) are surfaced explicitly.

---

## Key Design Decisions

- **YoE is always derived from `career_history`**, never from
  `profile.years_of_experience` — the latter is a deliberately unreliable field
  in this dataset, with planted over-claimers up to ~18× the real duration.
- **Notice period uses a gentle policy** (≤ 60 days no penalty) — the JD treats
  it as a mild demerit, not a dealbreaker.
- **Salary uses the top of the expected range**, not the midpoint, to avoid
  rewarding wide low-anchored bands.
- **Penalties are multiplicative** so stacked concerns compound, mirroring how a
  recruiter discounts a profile with several red flags.

---

## Performance & Compute Compliance

| Constraint | Limit | Actual |
|------------|-------|--------|
| Runtime | < 5 min (CPU) | **~26 s** (100K candidates) |
| Memory | < 16 GB | **~1.6 GB** |
| Network | None (offline) | **No external calls** |
| GPU | Not used | **CPU only** |
| Dependencies | — | **None** (pure Python stdlib) |
| Pre-computation | — | **None required** |

Time-based features use a fixed reference date (`REFERENCE_DATE = 2026-06-12`),
so scores are **fully reproducible** — running the pipeline twice produces an
identical CSV.

---

## Results

| Metric | Value |
|--------|-------|
| Top score (rank 1) | 0.7616 |
| 100th score | 0.4457 |
| Score spread | 0.3159 |
| Honeypots excluded | 81,546 / 100,000 |
| Honeypots in top 100 | 0 |
| Adjacent reasoning duplicates | 0 |

The submission passes the official `validate_submission.py` checks: 100 data
rows, ranks 1–100 each appearing once, scores non-increasing by rank, and
equal-score ties broken by `candidate_id` ascending.

---

## AI Tools Declaration

AI tools (**Claude**, **GitHub Copilot**) were used in the development workflow
for architecture discussion, code review, feature design, iterative testing, and
the reasoning engine. No candidate data was
sent to any external LLM. The ranking logic is entirely deterministic and runs
fully offline.
