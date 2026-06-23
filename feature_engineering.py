# feature_engineering.py - Layer 1 Feature Extraction
# Version tracked in git repository.
"""
Layer 1 — Feature Extraction
=============================
Extracts ~50 explicit features from each candidate profile using pre-compiled
regex patterns and cached text normalization for 100K-candidate performance.

All functions are pure (no LLM calls, no network), O(1) pattern matching.
"""

import re
import functools
from datetime import datetime

from config import (
    SERVICE_COMPANIES, MUST_HAVE_CONCEPTS, VECTOR_DBS, CORE_LANG, JD_TEXT,
    EVALUATION_TERMS, HYBRID_SEARCH_TERMS, BAD_DOMAINS, RESEARCH_SIGNAL_TERMS,
    CLOSED_SYSTEM_TERMS, FICTIONAL_COMPANIES, PRIMARY_CITY_PREFERENCE,
    SECONDARY_CITY_PREFERENCE, CULTURE_MISFIT_TERMS, NON_COMPETE_TERMS,
    CV_SPEECH_DOMAIN_TERMS, MANAGER_ONLY_TITLES, FRAMEWORK_ONLY_TERMS,
    RANKING_TITLE_KEYWORDS, PRE_LLM_TOOLS,
    GHOST_RESPONSE_RATE_THRESHOLD, GHOST_INACTIVE_DAYS_THRESHOLD,
)

# ────────────────────────────────────────────────────────────────────────────
# Text Utilities (cached / pre-compiled)
# ────────────────────────────────────────────────────────────────────────────

_PATTERN_CACHE = {}

SHIPPING_KEYWORDS = {
    "shipped", "deployed", "launched", "production", "scaled",
    "rolled out", "released", "delivered"
}
TECHNICAL_TITLES = {
    "engineer", "developer", "scientist", "architect", "programmer",
    "coder", "technical lead", "tech lead"
}
CODING_KEYWORDS = {
    "python", "code", "coding", "develop", "build", "deploy",
    "shipping", "shipped", "engineer", "implemented", "architecture"
}


@functools.lru_cache(maxsize=512)
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def _get_pattern(terms):
    key = frozenset(terms)
    if key not in _PATTERN_CACHE:
        escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
        pattern_str = r"\b(?:" + "|".join(escaped) + r")\b"
        _PATTERN_CACHE[key] = re.compile(pattern_str)
    return _PATTERN_CACHE[key]


def any_term_in_text(text: str, terms: set) -> bool:
    normalized = normalize_text(text)
    return _get_pattern(terms).search(normalized) is not None


def count_terms_in_text(text: str, terms: set) -> int:
    normalized = normalize_text(text)
    return len(set(_get_pattern(terms).findall(normalized)))


def token_contains(text: str, term: str) -> bool:
    normalized = normalize_text(text)
    return re.search(r"\b" + re.escape(term) + r"\b", normalized) is not None


# ────────────────────────────────────────────────────────────────────────────
# Helper: Build Candidate Text Blobs (cached per candidate_id)
# ────────────────────────────────────────────────────────────────────────────

def _full_text(candidate: dict) -> str:
    """Summary + all career descriptions + skill names."""
    parts = [candidate["profile"].get("summary", "")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(parts)


def _career_text(candidate: dict) -> str:
    """All career descriptions concatenated."""
    return " ".join(job.get("description", "") for job in candidate.get("career_history", []))


# ────────────────────────────────────────────────────────────────────────────
# POSITIVE FEATURE SCORES (each returns a float in [0, 1])
# ────────────────────────────────────────────────────────────────────────────

def calculate_retrieval_score(candidate: dict) -> float:
    """Core retrieval/ranking skill match from text evidence."""
    text = _full_text(candidate)
    retrieval_hits = count_terms_in_text(text, MUST_HAVE_CONCEPTS.union(HYBRID_SEARCH_TERMS))
    vector_hits = count_terms_in_text(text, VECTOR_DBS)
    shipping = calculate_shipping_score(candidate)
    score = min(0.25 + min(retrieval_hits, 5) * 0.12 + min(vector_hits, 3) * 0.12 + shipping * 0.20, 1.0)
    return score


def calculate_production_fit(candidate: dict) -> float:
    """Composite: vector DB + hybrid search + evaluation + shipping."""
    score = 0.0
    if has_vector_db_experience(candidate):
        score += 0.30
    if has_hybrid_search_experience(candidate):
        score += 0.25
    if has_evaluation_experience(candidate):
        score += 0.20
    if calculate_shipping_score(candidate) > 0.3:
        score += 0.25
    return min(score, 1.0)


def calculate_evaluation_score(candidate: dict) -> float:
    """Check for explicit evaluation frameworks (NDCG, MRR, precision@k, A/B)."""
    text = _full_text(candidate)
    hits = count_terms_in_text(text, EVALUATION_TERMS)
    return min(0.15 + min(hits, 5) * 0.17, 1.0)


def calculate_pre_llm_score(candidate: dict) -> float:
    """
    Pre-2022 experience mentioning search/ranking/retrieval keywords
    and classic ML tools like xgboost, lightgbm, elasticsearch.
    Returns 0..1 scaled by depth.
    """
    history = candidate.get("career_history", [])
    core_kws = MUST_HAVE_CONCEPTS.union(VECTOR_DBS).union(HYBRID_SEARCH_TERMS).union(PRE_LLM_TOOLS)
    total_pre_months = 0
    tool_hits = 0
    for job in history:
        s_str = job.get("start_date")
        if not s_str:
            continue
        try:
            s_dt = datetime.strptime(s_str, "%Y-%m-%d")
        except Exception:
            continue
        if s_dt.year < 2022:
            title = job.get("title", "")
            desc = job.get("description", "")
            combined = title + " " + desc
            if any_term_in_text(combined, core_kws):
                total_pre_months += job.get("duration_months", 0)
                tool_hits += count_terms_in_text(combined, PRE_LLM_TOOLS)

    if total_pre_months == 0:
        return 0.0
    # Scale: 12 months → 0.5, 24+ months → 0.8, with tool depth bonus
    month_score = min(total_pre_months / 30.0, 0.8)
    tool_bonus = min(tool_hits * 0.05, 0.2)
    return min(month_score + tool_bonus, 1.0)


def calculate_ai_yoe(candidate: dict) -> float:
    """Calculate total years where role involved AI/ML/search keywords."""
    history = candidate.get("career_history", [])
    ai_months = 0
    core_kws = MUST_HAVE_CONCEPTS.union(VECTOR_DBS).union(HYBRID_SEARCH_TERMS)
    for job in history:
        title = job.get("title", "")
        desc = job.get("description", "")
        combined = title + " " + desc
        if any_term_in_text(combined, core_kws):
            ai_months += job.get("duration_months", 0)
    return ai_months / 12.0


def calculate_ai_yoe_score(candidate: dict) -> float:
    """Normalize AI YoE to [0, 1]."""
    ai_yoe = calculate_ai_yoe(candidate)
    if ai_yoe >= 5.0:
        return 1.0
    if ai_yoe >= 3.0:
        return 0.7 + 0.3 * (ai_yoe - 3.0) / 2.0
    if ai_yoe >= 1.0:
        return 0.3 + 0.4 * (ai_yoe - 1.0) / 2.0
    return 0.15


def calculate_ranking_yoe_score(candidate: dict) -> float:
    """
    Years specifically in ranking/search/retrieval roles (title-based).
    More precise than generic AI YoE.
    """
    history = candidate.get("career_history", [])
    ranking_months = 0
    for job in history:
        title = job.get("title", "")
        if any_term_in_text(title, RANKING_TITLE_KEYWORDS):
            ranking_months += job.get("duration_months", 0)
    ranking_yoe = ranking_months / 12.0
    if ranking_yoe >= 4.0:
        return 1.0
    if ranking_yoe >= 2.0:
        return 0.6 + 0.4 * (ranking_yoe - 2.0) / 2.0
    if ranking_yoe >= 0.5:
        return 0.2 + 0.4 * (ranking_yoe - 0.5) / 1.5
    return 0.1 if ranking_yoe > 0 else 0.0


def calculate_experience_fit(candidate: dict) -> float:
    """YOE sweet-spot scoring: 5-9 years is ideal for a Senior AI Engineer."""
    yoe = candidate["profile"].get("years_of_experience", 0)
    ai_yoe = calculate_ai_yoe(candidate)
    ratio = ai_yoe / yoe if yoe > 0 else 0.0

    if 5.0 <= yoe <= 9.0:
        base = 1.0
    elif yoe < 5.0:
        base = max(0.3, yoe / 5.0)
    else:
        base = max(0.5, 1.0 - (yoe - 9.0) / 15.0)

    # Over-senior with low AI ratio gets a discount
    if yoe > 9.0 and ratio < 0.5:
        base *= 0.75
    return base


def calculate_recent_coder_score(candidate: dict) -> float:
    """
    Evidence of hands-on coding within the last 18 months.
    Returns 1.0 for strong evidence, 0.45 for none.
    """
    signals = candidate.get("redrob_signals", {})
    github = signals.get("github_activity_score", -1)
    if github > 30:
        return 1.0

    history = candidate.get("career_history", [])
    current_year = datetime.now().year
    for job in history:
        end_date = job.get("end_date")
        recent = False
        if not end_date:
            recent = True
        else:
            try:
                e_dt = datetime.strptime(end_date, "%Y-%m-%d")
                cutoff = datetime.now().replace(year=current_year - 1, month=6)
                recent = e_dt >= cutoff
            except Exception:
                pass
        if not recent:
            continue
        title = job.get("title", "")
        desc = job.get("description", "")
        if any_term_in_text(title, TECHNICAL_TITLES) or any_term_in_text(desc, CODING_KEYWORDS):
            return 1.0

    # Marginal github evidence
    if github > 0:
        return 0.7

    return 0.45


def calculate_location_score(candidate: dict) -> float:
    """Geography preference: Pune/Noida > metro > relocatable > other."""
    location = candidate["profile"].get("location", "").lower()
    normalized = normalize_text(location)
    if any(city in normalized for city in PRIMARY_CITY_PREFERENCE):
        return 1.0
    if any(city in normalized for city in SECONDARY_CITY_PREFERENCE):
        return 0.85
    if candidate["redrob_signals"].get("willing_to_relocate", False):
        return 0.75
    return 0.55


def calculate_notice_score(candidate: dict) -> float:
    """Notice period scoring: <=30 days is best, >90 is worst."""
    notice = candidate["redrob_signals"].get("notice_period_days", 90)
    if notice <= 15:
        return 1.0
    if notice <= 30:
        return 0.90
    if notice <= 45:
        return 0.75
    if notice <= 60:
        return 0.60
    if notice <= 90:
        return 0.40
    return 0.20


def calculate_response_score(candidate: dict) -> float:
    """
    Combined recruiter response rate, response time, activity recency,
    and open-to-work flag.
    """
    signals = candidate["redrob_signals"]
    response_rate = signals.get("recruiter_response_rate", 0.0)
    open_flag = float(signals.get("open_to_work_flag", False))

    # Response time: lower is better (0-72 hrs ideal)
    avg_resp_hrs = signals.get("avg_response_time_hours", 72)
    time_score = max(0.0, 1.0 - (avg_resp_hrs / 168.0))  # 168 hrs = 1 week

    # Activity recency
    last_active = signals.get("last_active_date", "")
    recency_score = 0.5
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_ago = (datetime.now() - last_dt).days
            recency_score = max(0.0, 1.0 - days_ago / 180.0)
        except Exception:
            pass

    return (response_rate * 0.30 + open_flag * 0.25 + time_score * 0.20 + recency_score * 0.25)


def calculate_shipping_score(candidate: dict) -> float:
    """Track record of deploying/shipping production systems."""
    history = candidate.get("career_history", [])
    score = 0.0
    current_year = datetime.now().year

    for job in history:
        desc = job.get("description", "").lower()
        end_date = job.get("end_date")
        end_year = current_year
        if end_date:
            try:
                end_year = datetime.strptime(end_date, "%Y-%m-%d").year
            except Exception:
                end_year = current_year

        years_ago = current_year - end_year
        weight = 3.0 if years_ago <= 2 else 1.5 if years_ago <= 5 else 0.7
        matches = sum(1 for kw in SHIPPING_KEYWORDS if kw in desc)
        score += matches * weight

    return min(score / 10.0, 1.0)


# ────────────────────────────────────────────────────────────────────────────
# BOOLEAN FEATURE CHECKS (used by scoring and reasoning)
# ────────────────────────────────────────────────────────────────────────────

def has_vector_db_experience(candidate: dict) -> bool:
    return any_term_in_text(_full_text(candidate), VECTOR_DBS)


def has_hybrid_search_experience(candidate: dict) -> bool:
    text = candidate["profile"].get("summary", "") + " " + _career_text(candidate)
    return any_term_in_text(text, HYBRID_SEARCH_TERMS)


def has_evaluation_experience(candidate: dict) -> bool:
    text = candidate["profile"].get("summary", "") + " " + _career_text(candidate)
    return any_term_in_text(text, EVALUATION_TERMS)


def has_pre_llm_experience(candidate: dict) -> bool:
    return calculate_pre_llm_score(candidate) > 0.0


def is_researcher_profile(candidate: dict) -> bool:
    text = candidate["profile"].get("summary", "") + " " + _career_text(candidate)
    return any_term_in_text(text, RESEARCH_SIGNAL_TERMS)


def has_closed_system_signal(candidate: dict) -> bool:
    text = candidate["profile"].get("summary", "") + " " + _career_text(candidate)
    return any_term_in_text(text, CLOSED_SYSTEM_TERMS)


def is_fictional_company_history(candidate: dict) -> bool:
    text = " ".join(job.get("company", "") for job in candidate.get("career_history", [])) + \
           " " + candidate["profile"].get("current_company", "")
    return any_term_in_text(text, FICTIONAL_COMPANIES)


def is_recent_coder(candidate: dict) -> bool:
    """Backwards-compatible boolean wrapper."""
    return calculate_recent_coder_score(candidate) >= 0.7


# ────────────────────────────────────────────────────────────────────────────
# DEALBREAKER / PENALTY DETECTORS (each returns True if the penalty applies)
# ────────────────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> bool:
    """Fictional companies, YoE mismatch, keyword-stuffed summary."""
    if is_fictional_company_history(candidate):
        return True

    skills = candidate.get("skills", [])
    exp_zero = sum(1 for s in skills if s.get("proficiency") in {"expert", "advanced"} and s.get("duration_months", 0) == 0)
    if exp_zero >= 1:
        return True

    profile = candidate["profile"]
    yoe = profile.get("years_of_experience", 0)
    total_months = sum(job.get("duration_months", 0) for job in candidate.get("career_history", []))
    sum_yoe = total_months / 12.0
    if abs(yoe - sum_yoe) >= 1.0:
        return True

    summary = profile.get("summary", "")
    if any_term_in_text(summary, MUST_HAVE_CONCEPTS.union(VECTOR_DBS)) and \
       not any_term_in_text(summary, {"engineer", "developer", "scientist", "architect", "programmer"}):
        return True

    return False


def has_non_compete(candidate: dict) -> bool:
    """Legal risk: non-compete clauses in profile/career text."""
    text = _full_text(candidate)
    return any_term_in_text(text, NON_COMPETE_TERMS)


def is_cv_speech_domain(candidate: dict) -> bool:
    """Primary domain is computer vision/speech/robotics — misaligned with search/ranking."""
    text = _full_text(candidate)
    # Only flag if CV/speech terms appear AND ranking terms do NOT
    cv_hits = count_terms_in_text(text, CV_SPEECH_DOMAIN_TERMS)
    ranking_hits = count_terms_in_text(text, MUST_HAVE_CONCEPTS.union(HYBRID_SEARCH_TERMS))
    return cv_hits >= 3 and ranking_hits <= 1


def is_consultancy_only(candidate: dict) -> bool:
    """Entire career at service/consultancy companies."""
    history = candidate.get("career_history", [])
    if not history:
        return True
    return all(
        any(service in job.get("company", "").lower() for service in SERVICE_COMPANIES)
        for job in history
    )


def is_job_hopper(candidate: dict) -> bool:
    """Average tenure under 18 months across career."""
    history = candidate.get("career_history", [])
    if len(history) < 2:
        return False
    durations = [job.get("duration_months", 0) for job in history if job.get("duration_months", 0) > 0]
    if not durations:
        return False
    avg_months = sum(durations) / len(durations)
    return avg_months < 18


def is_manager_only(candidate: dict) -> bool:
    """
    Title contains Manager/Director/VP AND lacks recent coding hands-on evidence.
    """
    current_title = normalize_text(candidate["profile"].get("current_title", ""))
    if not any_term_in_text(current_title, MANAGER_ONLY_TITLES):
        return False
    # Check if recent career has coding evidence
    return not is_recent_coder(candidate)


def is_framework_enthusiast(candidate: dict) -> bool:
    """
    Has LangChain/LlamaIndex but lacks evaluation frameworks.
    These candidates know the wrapper but not the underlying ranking science.
    """
    text = _full_text(candidate)
    has_framework = any_term_in_text(text, FRAMEWORK_ONLY_TERMS)
    has_eval = any_term_in_text(text, EVALUATION_TERMS)
    return has_framework and not has_eval


def is_ghost(candidate: dict) -> bool:
    """
    Active search flags where response rate is low and inactive days are high.
    """
    signals = candidate["redrob_signals"]
    response_rate = signals.get("recruiter_response_rate", 0.0)
    last_active = signals.get("last_active_date", "")

    inactive_days = 999
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            inactive_days = (datetime.now() - last_dt).days
        except Exception:
            pass

    return (response_rate < GHOST_RESPONSE_RATE_THRESHOLD and
            inactive_days > GHOST_INACTIVE_DAYS_THRESHOLD)


def has_flight_risk(candidate: dict) -> bool:
    """Offer acceptance rate below 0.20 — candidate collects offers but doesn't join."""
    oar = candidate["redrob_signals"].get("offer_acceptance_rate", -1)
    if oar < 0:  # -1 means no offer history, not a risk
        return False
    return oar < 0.20


def is_culture_misfit(candidate: dict) -> bool:
    """
    Candidate explicitly asks for stability, predictable schedules, mature codebases
    when the job demands high ambiguity and shipping velocity.
    """
    text = candidate["profile"].get("summary", "") + " " + _career_text(candidate)
    return any_term_in_text(text, CULTURE_MISFIT_TERMS)


def is_research_only(candidate: dict) -> bool:
    """Research-heavy profile with no shipping evidence."""
    return is_researcher_profile(candidate) and calculate_shipping_score(candidate) <= 0.2


# ────────────────────────────────────────────────────────────────────────────
# STABILITY (used as a soft modifier, not a hard penalty)
# ────────────────────────────────────────────────────────────────────────────

def calculate_stability_score(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    if not history:
        return 1.0
    durations = [job.get("duration_months", 0) for job in history if job.get("duration_months", 0) > 0]
    if not durations:
        return 0.4
    avg_years = (sum(durations) / len(durations)) / 12.0
    if avg_years >= 3.0:
        return 1.0
    if avg_years >= 1.5:
        return 0.7 + 0.3 * (avg_years - 1.5) / 1.5
    return max(0.3, avg_years / 1.5)


# ────────────────────────────────────────────────────────────────────────────
# COMPLETENESS (profile quality)
# ────────────────────────────────────────────────────────────────────────────

def calculate_completeness_score(candidate: dict) -> float:
    critical = ["years_of_experience", "current_title", "current_company", "location", "summary"]
    filled = 0
    for k in critical:
        val = candidate["profile"].get(k)
        if val is not None and (not isinstance(val, str) or val.strip() != ""):
            filled += 1
    return filled / len(critical)


# ────────────────────────────────────────────────────────────────────────────
# EXTRACT FACTS (for reasoning / display)
# ────────────────────────────────────────────────────────────────────────────

def extract_facts(candidate: dict) -> dict:
    profile = candidate["profile"]
    history = candidate.get("career_history", [])
    skills = [s["name"].lower() for s in candidate.get("skills", [])]

    matched_skills = [s.title() for s in skills if any(
        concept in s for concept in MUST_HAVE_CONCEPTS.union(VECTOR_DBS).union(CORE_LANG)
    )]
    top_skills = matched_skills[:4] if matched_skills else [s.title() for s in skills[:4]]

    has_product_experience = any(
        not any(service in job["company"].lower() for service in SERVICE_COMPANIES)
        for job in history
    )
    best_product_company = next(
        (job["company"] for job in history
         if not any(service in job["company"].lower() for service in SERVICE_COMPANIES)),
        profile.get("current_company", "Unknown")
    )

    return {
        "yoe": profile.get("years_of_experience", 0),
        "current_title": profile.get("current_title", "Unknown"),
        "current_company": profile.get("current_company", "Unknown"),
        "top_skills": top_skills,
        "has_product_experience": has_product_experience,
        "best_product_company": best_product_company,
        "notice_period": candidate["redrob_signals"].get("notice_period_days", 90),
        "willing_to_relocate": candidate["redrob_signals"].get("willing_to_relocate", False),
        "location": profile.get("location", "Unknown"),
        "has_vector_db_experience": has_vector_db_experience(candidate),
        "has_evaluation_experience": has_evaluation_experience(candidate),
        "has_hybrid_search_experience": has_hybrid_search_experience(candidate),
        "has_pre_llm_experience": has_pre_llm_experience(candidate),
        "is_consultancy_only": is_consultancy_only(candidate),
        "is_researcher": is_researcher_profile(candidate),
        "has_closed_system_signal": has_closed_system_signal(candidate),
        "shipping_score": calculate_shipping_score(candidate),
    }
