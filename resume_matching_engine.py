import math

# ─── SKILL ALIASES (exact as provided) ───────────────────────────────────────
SKILL_ALIASES = {
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css",
    "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

# Precompute multi-word keys once, longest first
MULTI_KEYS = sorted([k for k in SKILL_ALIASES if " " in k], key=len, reverse=True)

# ─── RESUME DATASET ───────────────────────────────────────────────────────────
RESUMES = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

# ─── JOB DESCRIPTIONS ────────────────────────────────────────────────────────
JDS = [
    ("JD-1", "Kakao (ML Engineer)",
     "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
     "NLP, BERT, Feature Engineering, Statistics"),
    ("JD-2", "Naver (Backend Engineer)",
     "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
     "REST API, CI/CD, Redis"),
    ("JD-3", "Line (Frontend Engineer)",
     "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
     "Node.js, GraphQL, Redux, Jest, AWS"),
]


# ─── STEP 1 & 2: NORMALIZE + DEDUPLICATE ─────────────────────────────────────
def normalize_skills(raw):
    tokens = [t.strip() for t in raw.lower().split(",")]
    seen, result = set(), []
    for token in tokens:
        matched = None
        for mk in MULTI_KEYS:          # exact multi-word match first
            if token == mk:
                matched = SKILL_ALIASES[mk]
                break
        if matched is None:
            matched = SKILL_ALIASES.get(token)
        if matched is not None and matched not in seen:
            seen.add(matched)
            result.append(matched)
    return result


# ─── STEP 3: BUILD VOCABULARY ─────────────────────────────────────────────────
normalized = [(name, normalize_skills(raw)) for name, raw in RESUMES]

vocab     = sorted({s for _, skills in normalized for s in skills})
word2idx  = {w: i for i, w in enumerate(vocab)}
V         = len(vocab)


# ─── STEP 4: COMPUTE TF-IDF ──────────────────────────────────────────────────
N_DOCS = 10

df  = {s: sum(1 for _, skills in normalized if s in skills) for s in vocab}
idf = {s: math.log(N_DOCS / df[s]) for s in vocab}   # ln(10/df), no smoothing

def tfidf_vector(skills):
    N, vec = len(skills), [0.0] * V
    for s in skills:
        vec[word2idx[s]] = (1.0 / N) * idf[s]        # TF = 1/N after dedup
    return vec

resume_vectors = [(name, tfidf_vector(skills)) for name, skills in normalized]


# ─── STEP 5: BUILD JD BINARY VECTORS ─────────────────────────────────────────
def jd_binary_vector(req_str, pref_str):
    vec = [0] * V
    for s in normalize_skills(req_str + ", " + pref_str):
        if s in word2idx:
            vec[word2idx[s]] = 1
    return vec

jd_vectors = [(jd_id, jd_name, jd_binary_vector(req, pref))
              for jd_id, jd_name, req, pref in JDS]


# ─── STEP 6: COSINE SIMILARITY & RANKING ─────────────────────────────────────
def cosine(a, b):
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

for jd_id, jd_name, jd_vec in jd_vectors:
    scores = sorted(
        [(name, cosine(rvec, jd_vec)) for name, rvec in resume_vectors],
        key=lambda x: (-x[1], x[0])       # desc score, asc name for ties
    )
    print(f"{jd_id} — {jd_name}")
    print(", ".join(f"{n}({s:.2f})" for n, s in scores[:3]))
    print()
