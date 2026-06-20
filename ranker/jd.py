"""Structured representation of the job description.

The JD ("Senior AI Engineer - Founding Team" at Redrob AI) is dense and written
to defeat naive keyword matching. We translate it once, by hand, into the
machine-usable signal vocabulary below. This is the "understanding" step: rather
than embedding the raw JD and hoping cosine similarity captures intent, we encode
what the JD *means* - the must-haves, the nice-to-haves, the explicit anti-signals,
and the hard disqualifiers it spells out.

The semantic layer (semantic.py) still embeds the JD prose so that "Tier 5"
candidates - people who built ranking/recsys systems without using the trendy
vocabulary - are surfaced by meaning, not by buzzword overlap.
"""
from __future__ import annotations

# A compact natural-language query used for the dense/TF-IDF semantic match.
# It deliberately describes the WORK, not the buzzwords, so that candidates who
# describe building search/ranking/recommendation systems in plain language score
# well even without listing fashionable skills.
JD_SEMANTIC_QUERY = (
    "Senior AI engineer who builds production search, retrieval, ranking and "
    "recommendation systems for real users at a product company. Works on "
    "embeddings based retrieval, vector search, hybrid search, semantic search, "
    "learning to rank, candidate matching and relevance. Strong Python engineer "
    "who ships code, designs evaluation frameworks for ranking quality using "
    "NDCG, MRR and MAP, runs A/B tests, and reasons about latency and quality "
    "tradeoffs. Comfortable with information retrieval and natural language "
    "processing, embeddings, transformers, fine tuning LLMs with LoRA and QLoRA, "
    "and vector databases such as FAISS, Milvus, Pinecone, Weaviate, Qdrant, "
    "Elasticsearch and OpenSearch. Applied machine learning in production, not "
    "pure research, with a scrappy ship-first product mindset."
)

# Titles that signal a hands-on engineering / applied-ML career. Matched as
# substrings (lower-cased) against current + historical titles.
POSITIVE_TITLES = {
    "machine learning engineer": 1.0,
    "ml engineer": 1.0,
    "ai engineer": 1.0,
    "applied scientist": 1.0,
    "research engineer": 0.92,
    "nlp engineer": 1.0,
    "search engineer": 1.0,
    "ranking engineer": 1.0,
    "recommendation": 0.95,
    "data scientist": 0.85,
    "ml scientist": 0.95,
    "machine learning scientist": 0.95,
    "software engineer": 0.78,
    "backend engineer": 0.72,
    "data engineer": 0.70,
    "platform engineer": 0.68,
    "staff engineer": 0.80,
    "principal engineer": 0.80,
    "senior engineer": 0.72,
    "founding engineer": 0.85,
    "computer vision engineer": 0.55,  # adjacent; penalised later if no NLP/IR
    "deep learning": 0.85,
}

# Titles that strongly indicate a non-engineering career. Their presence in the
# CURRENT title is the single most decisive anti-keyword-stuffer signal: a
# "Marketing Manager" with every AI skill listed is still not an AI engineer.
NEGATIVE_TITLES = {
    "marketing manager": 0.05,
    "marketing": 0.10,
    "hr manager": 0.04,
    "human resources": 0.05,
    "recruiter": 0.10,
    "talent acquisition": 0.08,
    "sales executive": 0.05,
    "sales manager": 0.06,
    "account manager": 0.10,
    "accountant": 0.04,
    "finance": 0.12,
    "content writer": 0.06,
    "copywriter": 0.06,
    "graphic designer": 0.05,
    "ux designer": 0.30,
    "civil engineer": 0.05,
    "mechanical engineer": 0.05,
    "electrical engineer": 0.12,
    "operations manager": 0.10,
    "customer support": 0.05,
    "customer success": 0.10,
    "project manager": 0.18,
    "program manager": 0.20,
    "business analyst": 0.30,
    "product manager": 0.35,
    "teacher": 0.06,
    "professor": 0.20,  # academic-only is a disqualifier; handled separately
}

# Domain evidence terms. We weight HISTORY/SUMMARY occurrences far higher than
# skill-list occurrences because prose descriptions of real work are much harder
# to fake than a comma-separated skills array.
CORE_DOMAIN_TERMS = {
    "retrieval": 1.0,
    "ranking": 1.0,
    "rank": 0.5,
    "recommendation": 1.0,
    "recommender": 1.0,
    "recsys": 1.0,
    "search": 0.85,
    "semantic search": 1.0,
    "vector search": 1.0,
    "hybrid search": 1.0,
    "information retrieval": 1.0,
    "learning to rank": 1.0,
    "learning-to-rank": 1.0,
    "embedding": 0.95,
    "embeddings": 0.95,
    "relevance": 0.85,
    "ndcg": 1.0,
    "mrr": 0.9,
    "personalization": 0.8,
    "matching": 0.6,
    "nlp": 0.8,
    "natural language": 0.8,
    "bm25": 0.95,
    "elasticsearch": 0.85,
    "opensearch": 0.85,
    "faiss": 0.95,
    "milvus": 0.9,
    "pinecone": 0.9,
    "weaviate": 0.9,
    "qdrant": 0.9,
    "rag": 0.8,
    "sentence-transformers": 0.95,
    "sentence transformers": 0.95,
    "bge": 0.7,
    "transformer": 0.6,
    "bert": 0.7,
}

# Nice-to-have / bonus terms (the JD's "we'd like you to have" list).
BONUS_TERMS = {
    "lora": 0.6,
    "qlora": 0.7,
    "peft": 0.6,
    "fine-tune": 0.5,
    "fine tune": 0.5,
    "fine-tuning": 0.5,
    "xgboost": 0.5,
    "learning to rank": 0.6,
    "a/b test": 0.6,
    "ab test": 0.5,
    "open source": 0.4,
    "open-source": 0.4,
    "distributed": 0.4,
    "inference optimization": 0.5,
    "hr-tech": 0.5,
    "recruiting": 0.4,
    "marketplace": 0.4,
}

# Skills (from the skills array) that map to the role. Used with a trust weight.
RELEVANT_SKILLS = {
    "nlp", "information retrieval", "semantic search", "vector search",
    "embeddings", "sentence transformers", "transformers", "bert",
    "fine-tuning llms", "lora", "qlora", "peft", "rag", "llm",
    "learning to rank", "ranking", "recommendation systems", "recsys",
    "elasticsearch", "opensearch", "faiss", "milvus", "pinecone", "weaviate",
    "qdrant", "python", "pytorch", "tensorflow", "xgboost", "spark",
    "machine learning", "deep learning", "mlops", "feature engineering",
    "statistical modeling", "search",
}

# Skills that indicate a CV/speech/robotics specialisation. If a candidate's
# relevant skills are dominated by these AND there is little NLP/IR evidence,
# the JD says they'd be "re-learning fundamentals" - we down-weight.
CV_SPEECH_ROBOTICS_SKILLS = {
    "image classification", "object detection", "image segmentation",
    "speech recognition", "tts", "text-to-speech", "asr", "gans",
    "computer vision", "opencv", "robotics", "slam", "lidar", "point cloud",
}

# Indian consulting / services firms. A career spent ENTIRELY at these is a
# disqualifier per the JD ("People who have only worked at consulting firms ...").
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "mindtree", "ltimindtree", "lti",
    "hcl", "hexaware", "mphasis", "birlasoft", "persistent systems",
    "deloitte", "kpmg", "pwc", "ernst", "ibm global services",
}

# Preferred locations (Indian metros) - JD prefers Pune/Noida, open to relocation
# from Tier-1 Indian cities.
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon",
    "gurugram", "bengaluru", "bangalore", "ncr", "navi mumbai", "greater noida",
    "chennai", "kolkata", "ahmedabad",
}

# Experience band the JD describes (5-9 ideal, 4-10 acceptable).
EXP_IDEAL_LOW = 5.0
EXP_IDEAL_HIGH = 9.0
EXP_SOFT_LOW = 3.5
EXP_SOFT_HIGH = 11.0

# Component weights (sum to 1.0). Domain evidence dominates because it is the
# hardest signal to fake and the truest expression of "who actually fits".
WEIGHTS = {
    "domain": 0.30,
    "title": 0.22,
    "skills": 0.15,
    "experience": 0.10,
    "product": 0.10,
    "location": 0.08,
    "education": 0.05,
}
