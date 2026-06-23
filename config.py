import re

# 1. Strict Pure Services Penalty List (Lowercase for matching)
SERVICE_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "larsen & toubro", "mphasis", "hexaware"
}

# 2. Core JD Requirements for Heuristic Scoring
MUST_HAVE_CONCEPTS = {
    "retrieval", "ranking", "search", "recommendation", "matching", "vector", "semantic", "query",
    "machine learning", "ml", "llm", "embedding", "rag"
}
VECTOR_DBS = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vespa"}
CORE_LANG = {"python"}
EVALUATION_TERMS = {"ndcg", "mrr", "map", "offline evaluation", "online evaluation", "ab test", "ab testing", "evaluation framework", "metrics", "experiment", "precision", "recall", "f1"}
HYBRID_SEARCH_TERMS = {"hybrid search", "semantic search", "dense retrieval", "sparse retrieval", "bm25", "vector search", "fusion search"}

BAD_DOMAINS = {
    "computer vision", "cv", "speech", "robotics", "autonomous", "self-driving", "image classification", "gesture", "vision"
}
RESEARCH_SIGNAL_TERMS = {
    "research", "paper", "publication", "iclr", "neurips", "arxiv", "researcher", "academic", "thesis", "state of the art", "novel", "propose"
}
CLOSED_SYSTEM_TERMS = {
    "proprietary", "closed system", "in-house", "legacy system", "black box", "confidential", "internal platform", "enterprise only"
}
FICTIONAL_COMPANIES = {
    "dunder mifflin", "stark industries", "wayne enterprises", "acme corp", "hooli", "pied piper", "initech", "globex inc", "umbrella corp", "oscorp"
}

PREFERRED_CITIES = {"pune", "noida", "gurgaon", "gurugram", "delhi", "ncr", "mumbai", "hyderabad", "bangalore", "bengaluru"}
PRIMARY_CITY_PREFERENCE = {"pune", "noida"}
SECONDARY_CITY_PREFERENCE = {"delhi", "ncr", "mumbai", "hyderabad", "bangalore", "bengaluru"}

# --- NEW: Cultural Fit Mismatch Patterns ---
# Signals that a candidate prefers stability/predictability when the JD demands
# high ambiguity and shipping velocity
CULTURE_MISFIT_TERMS = {
    "stable environment", "predictable schedule", "clear specification", "clear specs",
    "well-defined process", "mature codebase", "established team", "work-life balance",
    "no overtime", "structured environment", "legacy maintenance", "slow-paced",
    "documentation first", "waterfall", "not startup"
}

# --- NEW: Non-Compete / Legal Risk Patterns ---
NON_COMPETE_TERMS = {
    "non-compete", "non compete", "noncompete", "restrictive covenant",
    "garden leave", "gardening leave", "binding agreement", "exclusivity clause"
}

# --- NEW: CV/Speech/Robotics Domain Mismatch Patterns ---
CV_SPEECH_DOMAIN_TERMS = {
    "computer vision", "image segmentation", "object detection", "yolo", "resnet",
    "convolutional neural", "speech recognition", "asr", "tts", "text to speech",
    "robotics", "autonomous driving", "self-driving", "lidar", "slam",
    "gesture recognition", "pose estimation", "image classification", "opencv"
}

# --- NEW: Manager-only / no-coding titles ---
MANAGER_ONLY_TITLES = {
    "director", "vp", "vice president", "head of", "chief", "cto", "ceo",
    "program manager", "project manager", "delivery manager", "engagement manager",
    "practice head", "group manager"
}

# --- NEW: Framework Enthusiast without eval (LangChain-only risk) ---
FRAMEWORK_ONLY_TERMS = {
    "langchain", "llamaindex", "llama index", "autogen", "crewai",
    "haystack", "semantic kernel"
}

# --- NEW: Ranking-specific job title keywords ---
RANKING_TITLE_KEYWORDS = {
    "search", "ranking", "retrieval", "recommendation", "relevance",
    "discovery", "matching", "information retrieval", "ir engineer"
}

# --- NEW: Pre-LLM era classic ML tools ---
PRE_LLM_TOOLS = {
    "xgboost", "lightgbm", "catboost", "sklearn", "scikit-learn",
    "gradient boosting", "random forest", "logistic regression",
    "feature engineering", "elasticsearch", "solr", "lucene",
    "learning to rank", "ltr", "lambdamart"
}

JD_TEXT = """
Senior AI Engineer building ranking, retrieval, and matching systems.
Needs production experience owning retrieval and ranking systems, embeddings-based search, hybrid search, and vector database deployment.
Must be comfortable solving real user problems and shipping product-grade systems under time constraints.
Candidate should have strong Python fluency, evaluation framework experience, and a product orientation over research.
"""

# 4. Multipliers & Thresholds
GHOST_RESPONSE_RATE_THRESHOLD = 0.25
GHOST_INACTIVE_DAYS_THRESHOLD = 60
NOTICE_PERIOD_PREFERRED = 60
IDEAL_YOE_MIN = 5.0
IDEAL_YOE_MAX = 9.0
IDEAL_AI_YOE = 5.0
MAX_NOTICE_DAYS = 90

# 5. Layer 2 Scoring Weights (positive features, sum to ~1.0)
POSITIVE_WEIGHTS = {
    "retrieval_score":     0.18,   # Core retrieval/ranking skill match
    "production_fit":      0.16,   # Vector DB + hybrid search + eval + shipping
    "evaluation_score":    0.12,   # NDCG/MRR/precision@k evidence
    "pre_llm_score":       0.10,   # Pre-2022 search/ranking/ML experience
    "ai_yoe_score":        0.08,   # Total AI/ML years of experience
    "ranking_yoe_score":   0.08,   # Years specifically in ranking/search roles
    "experience_fit":      0.06,   # Overall YOE sweet-spot (5-9 years)
    "recent_coder_score":  0.06,   # Evidence of hands-on coding within 18 months
    "location_score":      0.05,   # Geography preference (Pune/Noida > other)
    "notice_score":        0.04,   # Notice period (<=30 best)
    "response_score":      0.04,   # Recruiter response rate + activity recency
    "shipping_score":      0.03,   # Track record of shipping/deploying
}

# 6. Multiplicative Penalty Factors (dealbreakers, each in [0, 1])
PENALTY_DEFAULTS = {
    "honeypot":              0.0,   # Fictional company / YOE mismatch / keyword-stuffed
    "non_compete":           0.15,  # Active non-compete clause
    "cv_domain_mismatch":    0.30,  # Primary domain is CV/speech/robotics
    "consultancy_only":      0.50,  # All career at service companies
    "is_job_hopper":         0.55,  # Average tenure < 18 months
    "is_manager_only":       0.40,  # Manager/Director title, no recent coding
    "is_framework_enthusiast": 0.60,  # LangChain but no eval frameworks
    "is_ghost":              0.35,  # Low response rate + high inactive days
    "culture_misfit":        0.50,  # Stability-seeking in a high-velocity role
    "flight_risk":           0.55,  # Offer acceptance rate < 0.20
    "research_only":         0.60,  # Research-heavy with no shipping evidence
}
