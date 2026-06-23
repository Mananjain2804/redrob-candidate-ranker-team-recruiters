# reasoning.py - Factual Reasoning Sentence Generator
# Version tracked in git repository.
import hashlib
import random


def _hash_seed(cid: str) -> int:
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) % (2**32)


def generate_reasoning(scored_candidate: dict, last_template_type: str = "") -> tuple[str, str]:
    from config import VECTOR_DBS, EVALUATION_TERMS, HYBRID_SEARCH_TERMS, MUST_HAVE_CONCEPTS
    from feature_engineering import (
        is_honeypot, calculate_ai_yoe, is_consultancy_only,
        is_researcher_profile, is_recent_coder, is_fictional_company_history,
        calculate_shipping_score, has_non_compete, is_cv_speech_domain,
        is_job_hopper, is_manager_only, is_framework_enthusiast, is_ghost,
        has_flight_risk, is_culture_misfit
    )

    facts = scored_candidate["facts"]
    score = scored_candidate["score"]
    cid = scored_candidate["candidate_id"]
    candidate = scored_candidate.get("raw_candidate") or scored_candidate.get("raw")
    
    rng = random.Random(_hash_seed(cid))
    
    # 1. Flag Honeypots with highly specific reasons
    if is_honeypot(candidate):
        reasons = []
        if is_fictional_company_history(candidate):
            history = candidate.get("career_history", [])
            companies = [job.get("company", "") for job in history] + [candidate["profile"].get("current_company", "")]
            fictional = [c for c in companies if c.lower() in [
                "dunder mifflin", "stark industries", "wayne enterprises", "acme corp", "hooli", 
                "pied piper", "initech", "globex inc", "umbrella corp", "oscorp"
            ]]
            company_str = f" ({fictional[0]})" if fictional else ""
            reasons.append(f"presence of fictional company history{company_str}")
        
        profile = candidate["profile"]
        yoe = profile.get("years_of_experience", 0)
        total_months = sum(job.get("duration_months", 0) for job in candidate.get("career_history", []))
        sum_yoe = total_months / 12.0
        if abs(yoe - sum_yoe) >= 1.0:
            reasons.append(f"inconsistency between stated experience ({yoe} YoE) and career history duration ({sum_yoe:.1f} years)")
            
        skills = candidate.get("skills", [])
        exp_zero_skills = [s.get("name", "") for s in skills if s.get("proficiency") in {"expert", "advanced"} and s.get("duration_months", 0) == 0]
        if exp_zero_skills:
            reasons.append(f"expert/advanced skills ({', '.join(exp_zero_skills[:2])}) listed with 0 months of usage")
            
        summary_clean = candidate["profile"].get("summary", "").lower()
        has_concepts = any(concept in summary_clean for concept in MUST_HAVE_CONCEPTS.union(VECTOR_DBS))
        has_titles = any(t in summary_clean for t in ["engineer", "developer", "scientist", "architect", "programmer"])
        if has_concepts and not has_titles:
            reasons.append("keyword-stuffed summary without technical titles or job roles")
            
        reason_str = ", ".join(reasons) if reasons else "suspicious profile inconsistency"
        reasoning = f"Profile flagged as suspicious/honeypot due to {reason_str}."
        return reasoning, "honeypot"

    # 2. Extract Specific facts (Zero hallucination)
    yoe = facts.get("yoe", 0)
    ai_yoe = calculate_ai_yoe(candidate)
    current_title = facts.get("current_title", "Engineer")
    current_company = facts.get("current_company", "Current Company")
    best_company = facts.get("best_product_company") or current_company
    
    # Skills list
    skills_names = [s["name"].lower() for s in candidate.get("skills", [])]
    summary_lower = candidate["profile"].get("summary", "").lower()
    history_desc = " ".join([job.get("description", "").lower() for job in candidate.get("career_history", [])])
    
    # Match vector DBs
    matched_vector_dbs = []
    for db in VECTOR_DBS:
        if db in skills_names or f" {db} " in f" {summary_lower} " or f" {db} " in f" {history_desc} ":
            matched_vector_dbs.append(db.title())
            
    # Match evaluation terms
    matched_eval = []
    for term in EVALUATION_TERMS:
        if term in skills_names or f" {term} " in f" {summary_lower} " or f" {term} " in f" {history_desc} ":
            matched_eval.append(term)
    eval_display = []
    if any(t in matched_eval for t in ["ndcg", "mrr", "map"]):
        eval_display.append("search metrics (NDCG/MRR)")
    if any(t in matched_eval for t in ["ab test", "ab testing"]):
        eval_display.append("A/B testing")
    if any(t in matched_eval for t in ["offline evaluation", "online evaluation", "evaluation framework"]):
        eval_display.append("evaluation frameworks")
    if not eval_display and matched_eval:
        eval_display.append(matched_eval[0])
        
    # Match hybrid search terms
    matched_hybrid = []
    for term in HYBRID_SEARCH_TERMS:
        if term in skills_names or f" {term} " in f" {summary_lower} " or f" {term} " in f" {history_desc} ":
            matched_hybrid.append(term)
    hybrid_display = []
    if "hybrid search" in matched_hybrid or "dense retrieval" in matched_hybrid or "sparse retrieval" in matched_hybrid:
        hybrid_display.append("hybrid retrieval (dense/sparse)")
    elif "vector search" in matched_hybrid or "semantic search" in matched_hybrid:
        hybrid_display.append("semantic vector search")
    elif "bm25" in matched_hybrid:
        hybrid_display.append("BM25 retrieval")
    if not hybrid_display and matched_hybrid:
        hybrid_display.append(matched_hybrid[0])
        
    # Standard top skills for detail listing
    top_skills = facts.get("top_skills", [])
    skills_str = ", ".join(top_skills[:3]) if top_skills else "machine learning"
    
    # Notice & Availability
    signals = candidate.get("redrob_signals", {})
    notice = signals.get("notice_period_days", 90)
    open_to_work = signals.get("open_to_work_flag", False)
    
    # Location description
    loc = facts.get("location", "Unknown").lower()
    is_primary_loc = any(city in loc for city in ["pune", "noida"])
    is_secondary_loc = any(city in loc for city in ["delhi", "ncr", "mumbai", "hyderabad", "bangalore", "bengaluru"])
    willing_to_relocate = signals.get("willing_to_relocate", False)
    
    location_note = ""
    if is_primary_loc:
        location_note = f"based in preferred location ({facts.get('location')})"
    elif willing_to_relocate:
        location_note = "willing to relocate to Pune/Noida"
    elif is_secondary_loc:
        location_note = f"located in {facts.get('location')}"
    else:
        location_note = f"based in {facts.get('location')}"

    # Build component sentences
    # Introduce candidate and role
    if score >= 0.70:
        template_type = "strong"
        intro_options = [
            f"Strong Senior AI Engineer with {yoe:.1f} YOE (including {ai_yoe:.1f} years in AI/ML), currently working as {current_title} at {best_company}.",
            f"High-caliber candidate with {yoe:.1f} years of production ML experience, showcasing solid leadership as {current_title} at {best_company}.",
            f"Excellent profile matching the JD's need for seniority, bringing {yoe:.1f} YOE and {ai_yoe:.1f} years focused on AI systems at {best_company}."
        ]
    elif score >= 0.45:
        template_type = "decent"
        intro_options = [
            f"Qualified ML candidate offering {yoe:.1f} YOE (with {ai_yoe:.1f} years in AI roles) and product-centric background at {best_company}.",
            f"Mid-to-senior profile with {yoe:.1f} YOE and experience as {current_title} at {best_company}, showing strong systems experience.",
            f"Solid candidate with {yoe:.1f} years of applied software and ML history, including product work at {best_company}."
        ]
    else:
        template_type = "weak"
        intro_options = [
            f"Candidate with {yoe:.1f} YOE and a background as {current_title} at {best_company}.",
            f"Adjacent profile showing {yoe:.1f} YOE with exposure to {skills_str}.",
            f"Software engineer profile with {yoe:.1f} years of experience, showing foundational understanding of ML concepts."
        ]
    intro_phrase = rng.choice(intro_options)

    # Technical depth connection (Vector DB, hybrid, evaluation)
    tech_points = []
    if matched_vector_dbs:
        tech_points.append(f"hands-on deployment of vector databases ({', '.join(matched_vector_dbs[:2])})")
    if hybrid_display:
        tech_points.append(f"practical knowledge of {hybrid_display[0]}")
    if eval_display:
        tech_points.append(f"familiarity with {eval_display[0]}")
        
    ship_score = calculate_shipping_score(candidate)
    if ship_score > 0.3:
        tech_points.append("a clear track record of shipping production-grade systems")
        
    if tech_points:
        if len(tech_points) >= 3:
            tech_phrase = f"Key strengths include {tech_points[0]}, {tech_points[1]}, and {tech_points[2]}."
        elif len(tech_points) == 2:
            tech_phrase = f"Key strengths include {tech_points[0]} and {tech_points[1]}."
        else:
            tech_phrase = f"Demonstrates {tech_points[0]} in past projects."
    else:
        tech_phrase = f"Technical toolkit includes skills in {skills_str}."

    # Availability & location sentence
    avail_parts = []
    if open_to_work:
        if notice <= 30:
            avail_parts.append(f"is immediately available with a short {notice}-day notice")
        elif notice <= 60:
            avail_parts.append(f"is actively looking with a manageable {notice}-day notice period")
        else:
            avail_parts.append(f"is open to work, but has a longer {notice}-day notice period")
    else:
        if notice <= 60:
            avail_parts.append(f"is available within a standard {notice}-day notice period")
        else:
            avail_parts.append(f"has a longer {notice}-day notice period")
            
    if location_note:
        avail_parts.append(f"is {location_note}")
        
    if len(avail_parts) == 2:
        avail_phrase = f"Candidate {avail_parts[0]} and {avail_parts[1]}."
    elif len(avail_parts) == 1:
        avail_phrase = f"Candidate {avail_parts[0]}."
    else:
        avail_phrase = "Candidate is available for opportunities."

    # Honest concern / Gap
    concerns = []
    if is_consultancy_only(candidate):
        concerns.append("consultancy-only career history which may lack product startup speed")
    if not matched_vector_dbs:
        concerns.append("limited explicit experience with vector databases in production")
    if not matched_eval:
        concerns.append("gaps in explicit offline/online evaluation metric design")
    if is_researcher_profile(candidate) and ship_score <= 0.2:
        concerns.append("a research-heavy background with sparse shipping evidence")
    if not is_recent_coder(candidate):
        concerns.append("limited evidence of recent hands-on coding activity")
    if notice > 75:
        concerns.append(f"a long notice period ({notice} days)")
    # New dealbreaker concerns
    if has_non_compete(candidate):
        concerns.append("potential legal risk from a non-compete clause")
    if is_cv_speech_domain(candidate):
        concerns.append("primary domain in CV/speech/robotics, misaligned with search/ranking")
    if is_job_hopper(candidate):
        concerns.append("short average tenure across roles (potential job-hopping pattern)")
    if is_manager_only(candidate):
        concerns.append("management-heavy title track with limited recent coding evidence")
    if is_framework_enthusiast(candidate):
        concerns.append("familiarity with LLM frameworks but lacking evaluation/ranking foundations")
    if is_ghost(candidate):
        concerns.append("low recruiter response rate and prolonged inactivity on platform")
    if has_flight_risk(candidate):
        concerns.append("historically low offer acceptance rate (flight risk)")
    if is_culture_misfit(candidate):
        concerns.append("signals preference for stable/predictable environments vs. high-velocity shipping")
        
    if concerns:
        filtered_concerns = [c for c in concerns if not ("notice" in c and "notice" in avail_phrase)]
        if filtered_concerns:
            if len(filtered_concerns) == 1:
                concern_phrase_options = [
                    f"A key gap to note is {filtered_concerns[0]}.",
                    f"The primary concern is {filtered_concerns[0]}.",
                    f"A minor concern is {filtered_concerns[0]}."
                ]
                concern_phrase = rng.choice(concern_phrase_options)
            else:
                concern_phrase_options = [
                    f"Potential gaps include {filtered_concerns[0]} and {filtered_concerns[1]}.",
                    f"Key concerns to address are {filtered_concerns[0]}, as well as {filtered_concerns[1]}.",
                    f"Areas of improvement include {filtered_concerns[0]} and {filtered_concerns[1]}."
                ]
                concern_phrase = rng.choice(concern_phrase_options)
        else:
            concern_phrase = "No major gaps were identified relative to the core requirements."
    else:
        concern_phrase = "No major gaps were identified relative to the core requirements."

    # Combine using 4 different stylistic narrative layouts to maximize variety
    layout_choice = rng.randint(0, 3)
    if layout_choice == 0:
        reasoning = f"{intro_phrase} {tech_phrase} {avail_phrase} {concern_phrase}"
    elif layout_choice == 1:
        reasoning = f"{intro_phrase} {tech_phrase} {concern_phrase} {avail_phrase}"
    elif layout_choice == 2:
        reasoning = f"{intro_phrase} {avail_phrase} {tech_phrase} {concern_phrase}"
    else:
        reasoning = f"{intro_phrase} {avail_phrase} {concern_phrase} Overall, they bring strong capabilities in {skills_str}."
        
    reasoning = " ".join(reasoning.split())
    return reasoning, template_type

