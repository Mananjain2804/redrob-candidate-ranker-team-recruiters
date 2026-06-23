# ranker.py - Layer 2 Scoring & Penalties Engine
# Version tracked in git repository.
"""
Layer 2 — Weighted Scoring & Multiplicative Penalties
=====================================================
Two-pass scoring:
  1. Base Score (positives): weighted sum of positive feature scores (max ~1.0)
  2. Penalty Multiplier (dealbreakers): product of all applicable penalty factors

Final Score = Base Score * Penalty Multiplier
Tie-breaking: candidate_id ascending (deterministic).
"""

from feature_engineering import (
    # Positive feature scorers
    calculate_retrieval_score,
    calculate_production_fit,
    calculate_evaluation_score,
    calculate_pre_llm_score,
    calculate_ai_yoe_score,
    calculate_ranking_yoe_score,
    calculate_experience_fit,
    calculate_recent_coder_score,
    calculate_location_score,
    calculate_notice_score,
    calculate_response_score,
    calculate_shipping_score,
    # Dealbreaker detectors
    is_honeypot,
    has_non_compete,
    is_cv_speech_domain,
    is_consultancy_only,
    is_job_hopper,
    is_manager_only,
    is_framework_enthusiast,
    is_ghost,
    has_flight_risk,
    is_culture_misfit,
    is_research_only,
    # Facts for reasoning
    extract_facts,
)

from config import POSITIVE_WEIGHTS, PENALTY_DEFAULTS


def score_candidate(c: dict) -> float:
    """
    Score a single candidate using the two-layer architecture:
    Layer 2a: Sum of weighted positive features
    Layer 2b: Multiplicative penalties for dealbreakers
    """

    # ── Layer 2b: Check hard dealbreakers first (fast exit) ──
    if is_honeypot(c):
        return 0.0

    # ── Layer 2a: Compute positive feature scores ──
    feature_scores = {
        "retrieval_score":     calculate_retrieval_score(c),
        "production_fit":      calculate_production_fit(c),
        "evaluation_score":    calculate_evaluation_score(c),
        "pre_llm_score":       calculate_pre_llm_score(c),
        "ai_yoe_score":        calculate_ai_yoe_score(c),
        "ranking_yoe_score":   calculate_ranking_yoe_score(c),
        "experience_fit":      calculate_experience_fit(c),
        "recent_coder_score":  calculate_recent_coder_score(c),
        "location_score":      calculate_location_score(c),
        "notice_score":        calculate_notice_score(c),
        "response_score":      calculate_response_score(c),
        "shipping_score":      calculate_shipping_score(c),
    }

    base_score = sum(
        feature_scores[feat] * POSITIVE_WEIGHTS[feat]
        for feat in POSITIVE_WEIGHTS
    )

    # ── Layer 2b: Multiplicative penalties ──
    penalty_multiplier = 1.0

    if has_non_compete(c):
        penalty_multiplier *= PENALTY_DEFAULTS["non_compete"]
    if is_cv_speech_domain(c):
        penalty_multiplier *= PENALTY_DEFAULTS["cv_domain_mismatch"]
    if is_consultancy_only(c):
        penalty_multiplier *= PENALTY_DEFAULTS["consultancy_only"]
    if is_job_hopper(c):
        penalty_multiplier *= PENALTY_DEFAULTS["is_job_hopper"]
    if is_manager_only(c):
        penalty_multiplier *= PENALTY_DEFAULTS["is_manager_only"]
    if is_framework_enthusiast(c):
        penalty_multiplier *= PENALTY_DEFAULTS["is_framework_enthusiast"]
    if is_ghost(c):
        penalty_multiplier *= PENALTY_DEFAULTS["is_ghost"]
    if is_culture_misfit(c):
        penalty_multiplier *= PENALTY_DEFAULTS["culture_misfit"]
    if has_flight_risk(c):
        penalty_multiplier *= PENALTY_DEFAULTS["flight_risk"]
    if is_research_only(c):
        penalty_multiplier *= PENALTY_DEFAULTS["research_only"]

    final_score = base_score * penalty_multiplier
    return round(final_score, 4)


def rank_candidates(candidates: list[dict], top_n: int = 100) -> list[dict]:
    """
    Score all candidates, sort by score descending with tie-breaking
    by candidate_id ascending, return top N.
    """
    scored = []
    for c in candidates:
        score = score_candidate(c)
        facts = extract_facts(c)
        scored.append({
            "candidate_id": c["candidate_id"],
            "score": score,
            "facts": facts,
            "raw": c,
            "raw_candidate": c,
        })

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return scored[:top_n]
