"""
config.py — All the domain knowledge for the Redrob Senior AI Engineer ranker.

Every weight, vocabulary, and threshold lives here so the scoring logic in
score.py stays readable and the whole system is auditable in one place.

The job description (job_description.md) is the source of truth. The constants
below are a structured encoding of *what the JD actually means*, not a keyword
list. See README.md Section "How the JD maps to features".
"""

# ---------------------------------------------------------------------------
# 1. The role, expressed as a dense semantic query for the TF-IDF retrieval lane
# ---------------------------------------------------------------------------
# This paragraph is matched (cosine, TF-IDF) against each candidate's full text.
# It is deliberately written in the *language a strong candidate would use* so
# that "plain-language" candidates who describe building retrieval/ranking
# systems without buzzwords still surface. It is NOT a keyword filter.
JD_QUERY = (
    "senior ai engineer building production machine learning systems for search "
    "retrieval and ranking. embeddings based retrieval, dense vector recall, hybrid "
    "search combining bm25 with semantic vector search. vector databases faiss "
    "pinecone weaviate qdrant milvus opensearch elasticsearch. embedding model "
    "selection and fine tuning, sentence transformers, bge, e5. learning to rank, "
    "recommendation systems, candidate job matching, relevance, re-ranking with "
    "llms, retrieval augmented generation. llm fine tuning lora qlora peft. "
    "evaluation frameworks for ranking ndcg mrr map offline online ab testing "
    "recruiter engagement. shipped end to end ranking search recommendation system "
    "to real users at scale at a product company. applied ml, nlp, information "
    "retrieval, strong python, mlops, model deployment, production experience, "
    "embedding drift, index refresh, retrieval quality regression."
)

# ---------------------------------------------------------------------------
# 2. Title taxonomy — how on-track is the candidate's *role*?
# ---------------------------------------------------------------------------
# The single strongest discriminator against keyword stuffers: a "Marketing
# Manager" with every AI skill listed is NOT a fit. Score is on [0, 1].
# Matched as substrings (lowercased) against current + historical titles.
TITLE_TIERS = {
    # Bullseye: AI/ML engineering on retrieval/ranking/search
    1.00: ["senior ai engineer", "lead ai engineer", "staff machine learning",
           "principal ai", "recommendation systems engineer", "search engineer",
           "senior applied scientist", "applied scientist"],
    0.92: ["ai engineer", "machine learning engineer", "ml engineer",
           "applied ml engineer", "senior machine learning", "nlp engineer",
           "senior nlp engineer", "ai research engineer", "research engineer"],
    0.80: ["data scientist", "senior data scientist", "ai specialist",
           "senior software engineer (ml)", "computer vision engineer"],
    # Adjacent: strong infra/data backbone the JD respects, but not the core role
    0.60: ["data engineer", "senior data engineer", "analytics engineer",
           "backend engineer", "ml platform", "machine learning platform"],
    0.40: ["software engineer", "senior software engineer", "full stack",
           "backend developer", "cloud engineer", "devops engineer",
           "data analyst", "junior ml engineer"],
    0.20: ["frontend", "mobile developer", "qa engineer", ".net developer",
           "java developer"],
}
# Anything not matched above (HR Manager, Accountant, Marketing Manager,
# Graphic Designer, Civil/Mechanical Engineer, Sales, Customer Support, etc.)
# gets this floor. These are the keyword-stuffer host bodies.
TITLE_DEFAULT = 0.05

# ---------------------------------------------------------------------------
# 3. Skill vocabulary — grouped by relevance to THIS role
# ---------------------------------------------------------------------------
# Skills only count when *corroborated* (duration + endorsements + proficiency),
# see score.py:skill_trust. This is what defeats lazy keyword stuffing.
CORE_SKILLS = {  # the "absolutely need" cluster from the JD
    "embeddings", "sentence transformers", "sentence-transformers", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "vector search", "vector databases", "information retrieval", "retrieval",
    "semantic search", "recommendation systems", "learning to rank",
    "ranking", "bm25", "haystack", "bge", "e5",
}
ML_SKILLS = {  # strong supporting ML depth
    "machine learning", "deep learning", "nlp", "natural language processing",
    "pytorch", "tensorflow", "scikit-learn", "hugging face transformers",
    "transformers", "fine-tuning llms", "lora", "qlora", "peft", "llms",
    "large language models", "mlops", "mlflow", "feature engineering",
    "xgboost", "lightgbm", "model deployment", "rag",
}
# The "nice to have but won't reject" cluster — small positive weight.
BONUS_SKILLS = {
    "kubernetes", "docker", "spark", "airflow", "kafka", "aws", "gcp", "azure",
    "distributed systems", "bentoml", "ray", "triton",
}
# Explicit anti-signals the JD calls out: recent-LangChain framework enthusiasts.
# Presence is fine; *over-reliance with no corroborated ML depth* is penalized.
FRAMEWORK_HYPE_SKILLS = {"langchain", "llamaindex", "prompt engineering", "autogen"}
# Off-track specialisms the JD down-weights when they DOMINATE without NLP/IR.
OFFTRACK_SKILLS = {
    "image classification", "object detection", "speech recognition", "tts",
    "diffusion models", "robotics", "slam", "computer vision",
}

# ---------------------------------------------------------------------------
# 4. Employer knowledge — product vs services
# ---------------------------------------------------------------------------
# The JD explicitly down-weights candidates whose ENTIRE career is at
# IT-services / consulting firms, and up-weights product-company experience.
SERVICES_COMPANIES = {
    "infosys", "tcs", "tata consultancy", "wipro", "accenture", "capgemini",
    "cognizant", "hcl", "mindtree", "tech mahindra", "lti", "ltimindtree",
    "mphasis", "hexaware", "birlasoft", "persistent",
}
PRODUCT_COMPANIES = {
    "swiggy", "zomato", "cred", "razorpay", "uber", "flipkart", "meesho",
    "phonepe", "ola", "dream11", "mad street den", "google", "meta",
    "facebook", "amazon", "microsoft", "apple", "netflix", "linkedin",
    "nvidia", "atlassian", "stripe", "airbnb", "sprinklr", "freshworks",
    "postman", "browserstack", "zeta", "navi",
}
# Fictional placeholder employers in the dataset — treated as neutral product.
NEUTRAL_COMPANIES = {
    "hooli", "pied piper", "globex inc", "globex", "initech", "acme corp",
    "wayne enterprises", "stark industries", "dunder mifflin",
}

# ---------------------------------------------------------------------------
# 5. Location — JD wants India / specific metros, or willing to relocate
# ---------------------------------------------------------------------------
TARGET_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon",
    "gurugram", "bangalore", "bengaluru", "delhi ncr", "ncr", "navi mumbai",
}

# ---------------------------------------------------------------------------
# 6. Component weights for the additive FIT score (before behavioral modifier)
# ---------------------------------------------------------------------------
# These sum to 1.0 and were tuned against the trap profiles in the public
# sample (target AI engineers rank high; stuffers / honeypots collapse).
WEIGHTS = {
    "title":       0.252,   # is the ROLE genuinely on-track (anti-stuffer #1)
    "skill_trust": 0.195,   # corroborated AI skills (anti-stuffer #2)
    "semantic":    0.208,   # hybrid BM25+TF-IDF JD match (RRF)
    "career":      0.223,   # built ranking/search/rec at a product company
    "experience":  0.057,   # 5-9 yrs sweet spot (soft)
    "education":   0.055,   # mild; JD cares little about pedigree
    "location":    0.01,   # India / target metro / relocate
}

# ---------------------------------------------------------------------------
# 7. Behavioral availability — applied as a MULTIPLIER, per redrob_signals_doc
# ---------------------------------------------------------------------------
# "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
# response rate is, for hiring purposes, not actually available."
# The multiplier ranges ~0.55 (ghost) to ~1.08 (highly engaged & available).
BEHAVIORAL = {
    "floor": 0.55,        # most an unavailable-but-perfect candidate is docked
    "ceil": 1.08,         # small boost for genuinely engaged candidates
}

# ---------------------------------------------------------------------------
# 8. Honeypot guard thresholds
# ---------------------------------------------------------------------------
HONEYPOT = {
    "expert_zero_dur_min": 3,   # >=3 "expert" skills claimed with 0 months used
    "career_span_slack_yrs": 7, # career timeline exceeding YoE by > this = impossible
    "role_over_yoe_slack_m": 24,# a single role longer than (YoE*12 + this) = impossible
    "penalty": 0.02,            # multiply final score by this (sinks them, not -inf,
                                # so a near-miss real candidate is never hard-deleted)
}

TOP_N = 100
