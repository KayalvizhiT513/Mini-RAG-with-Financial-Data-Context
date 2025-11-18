import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.sparse import load_npz, save_npz

# --------------------------------------------------
# FAST MODEL LOAD (local model, no internet, instant)
# --------------------------------------------------
MODEL_PATH = "./local_model"

if not os.path.exists(MODEL_PATH):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.save("./local_model")
else:
    print("⚡ Loading model from local path...")
    model = SentenceTransformer(MODEL_PATH)   # loads in <1 sec

# --------------------------------------------------
# Cache file paths
# --------------------------------------------------
EMB_CACHE = "embeddings_cache.npy"
FAISS_CACHE = "faiss_index.bin"
TFIDF_VECTORIZER_CACHE = "tfidf_vectorizer.pkl"
TFIDF_MATRIX_CACHE = "tfidf_matrix.npz"

# --------------------------------------------------
# Data
# --------------------------------------------------
faq_df = pd.read_csv("faqs.csv")
fund_df = pd.read_csv("funds.csv")

def fund_to_text(row):
    return (
        f"{row['fund_name']} description: {row.get('category','')}. "
        f"It has a 3-year CAGR of {row['cagr_3yr (%)']}%, "
        f"volatility {row['volatility (%)']}%, "
        f"Sharpe ratio {row['sharpe_ratio']}."
    )

fund_df["fund_text"] = fund_df.apply(fund_to_text, axis=1)

corpus = list(faq_df["question"] + " " + faq_df["answer"]) + list(fund_df["fund_text"])
corpus_type = ["faq"] * len(faq_df) + ["fund"] * len(fund_df)

# --------------------------------------------------
# Load or compute embeddings
# --------------------------------------------------

if os.path.exists(EMB_CACHE):
    print("⚡ Loading cached embeddings...")
    embeddings = np.load(EMB_CACHE)
else:
    print("⏳ Computing embeddings (first run only)...")
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMB_CACHE, embeddings)
    print("💾 Saved embeddings cache.")

# --------------------------------------------------
# Load or create FAISS index
# --------------------------------------------------
dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)

if os.path.exists(FAISS_CACHE):
    print("⚡ Loading cached FAISS index...")
    faiss.read_index(FAISS_CACHE)
else:
    print("⏳ Building FAISS index (first run only)...")
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_CACHE)
    print("💾 Saved FAISS index cache.")

# --------------------------------------------------
# Load or compute TF–IDF components
# --------------------------------------------------
if os.path.exists(TFIDF_VECTORIZER_CACHE) and os.path.exists(TFIDF_MATRIX_CACHE):
    print("⚡ Loading cached TF-IDF vectorizer + matrix...")
    with open(TFIDF_VECTORIZER_CACHE, "rb") as f:
        tfidf = pickle.load(f)

    tfidf_matrix = load_npz(TFIDF_MATRIX_CACHE)
else:
    print("⏳ Computing TF-IDF (first run only)...")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(corpus)

    with open(TFIDF_VECTORIZER_CACHE, "wb") as f:
        pickle.dump(tfidf, f)

    save_npz(TFIDF_MATRIX_CACHE, tfidf_matrix)
    print("💾 Saved TF-IDF cache.")

# --------------------------------------------------
# RAG Retrieval Functions
# --------------------------------------------------

def semantic_search(query, k=6):
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(q_emb, k)
    return [{"text": corpus[i], "type": corpus_type[i], "id": int(i)} for i in I[0]]

def lexical_search(query, k=8):
    q_vec = tfidf.transform([query])
    print(f"Query TF-IDF vector shape: {q_vec.shape}, corpus TF-IDF matrix shape: {tfidf_matrix.shape}")
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    idx = np.argsort(-sims)[:k]   # descending similarity
    return [{"text": corpus[i], "type": corpus_type[i], "id": int(i)} for i in idx]

def hybrid_search(query, k=8, alpha=0.5):
    sem = semantic_search(query)
    lex = lexical_search(query)
    scores = {}
    for rank, doc in enumerate(sem):
        scores[doc["id"]] = scores.get(doc["id"], 0) + alpha / (rank + 1)
    for rank, doc in enumerate(lex):
        scores[doc["id"]] = scores.get(doc["id"], 0) + (1 - alpha) / (rank + 1)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
    return [{"text": corpus[i], "type": corpus_type[i], "id": int(i)} for i, _ in ranked]

# retrieval mode controller
def retrieve_context(query, mode="semantic"):
    if mode == "lexical":
        return lexical_search(query)
    elif mode == "hybrid":
        return hybrid_search(query)
    return semantic_search(query)

# --------------------------------------------------
# RAG-Enhanced Intent Templates
# --------------------------------------------------

INTENT_TEMPLATES = {
    "definition": [
        "what is", "explain", "meaning of", "define", "help me understand"
    ],
    "comparison": [
        "which fund performs best", "top performing funds",
        "rank funds", "compare funds", "best cagr",
        "highest sharpe ratio", "best returns", "which fund"
    ],
    "generic": [
        "tell me about mutual funds", "describe this fund",
        "help me choose a fund"
    ]
}

intent_embeds = {
    label: model.encode(phrases, convert_to_numpy=True)
    for label, phrases in INTENT_TEMPLATES.items()
}

def classify_intent_semantically(query):
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    scores = {}
    for label, embeds in intent_embeds.items():
        sims = cosine_similarity([q_emb], embeds)[0]
        scores[label] = np.max(sims)
    return max(scores, key=scores.get)

def detect_intent(query):
    semantic_guess = classify_intent_semantically(query)
    q = query.lower()
    if any(w in q for w in ["what is", "define", "meaning"]):
        return "definition"
    if any(w in q for w in ["best", "top", "highest", "lowest", "rank", "compare"]):
        return "comparison"
    return semantic_guess

# --------------------------------------------------
# Metric Detection (RAG-Based)
# --------------------------------------------------

METRIC_TEMPLATES = {
    "cagr_3yr (%)": ["CAGR", "returns", "growth", "performance", "appreciation"],
    "sharpe_ratio": ["sharpe", "risk-adjusted return", "consistency"],
    "volatility (%)": ["volatility", "risk", "swings", "fluctuation"]
}

metric_embeds = {
    metric: model.encode(phrases, convert_to_numpy=True)
    for metric, phrases in METRIC_TEMPLATES.items()
}
 
def detect_metric(query):
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    scores = {}
    for metric, embeds in metric_embeds.items():
        sims = cosine_similarity([q_emb], embeds)[0]
        scores[metric] = np.max(sims)
    best = max(scores, key=scores.get)
    return best

# --------------------------------------------------
# Direction Detection
# --------------------------------------------------

DIRECTION_TEMPLATES = {
    "desc": ["best", "highest", "top", "maximize", "improve"],
    "asc": ["least", "lowest", "minimum", "reduce", "minimize"]
}

dir_embeds = {
    d: model.encode(phrases, convert_to_numpy=True)
    for d, phrases in DIRECTION_TEMPLATES.items()
}

def detect_direction(query):
    q = query.lower()

    # RULE-BASED OVERRIDE (Fixes your issue)
    if any(w in q for w in ["best", "highest", "top", "maximum"]):
        return "desc"
    if any(w in q for w in ["least", "lowest", "minimum", "bottom"]):
        return "asc"

    # FALLBACK — Semantic detection
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    scores = {}

    for d, embeds in dir_embeds.items():
        sims = cosine_similarity([q_emb], embeds)[0]
        scores[d] = np.max(sims)

    print(f"Direction scores: {scores} for query: {query}")
    return max(scores, key=scores.get)

# --------------------------------------------------
# Deterministic Ranking
# --------------------------------------------------

def rank_top_funds(metric, direction="desc", top_k=3):
    ascending = direction == "asc"
    df_sorted = fund_df.sort_values(by=metric, ascending=ascending)
    return df_sorted.head(top_k).to_dict(orient="records")

# --------------------------------------------------
# Full RAG Pipeline (ONLY RETURNED STRUCTURE CHANGED)
# --------------------------------------------------

def enhanced_rag_pipeline(query, retrieval_mode="semantic"):
    docs = retrieve_context(query, retrieval_mode)
    intent = detect_intent(query)
    metric = detect_metric(query)

    # 1 — Definition
    if intent == "definition":
        for d in docs:
            if d["type"] == "faq":
                return d["text"], docs, []

    # 2 — Comparison (TOP FUNDS)
    if intent == "comparison" and metric:
        direction = detect_direction(query)
        top_funds = rank_top_funds(metric, direction)

        answer = f"Top funds ranked by {metric} ({'highest' if direction=='desc' else 'lowest'} first):\n"
        for f in top_funds:
            answer += f"- {f['fund_name']} ({metric}: {f[metric]})\n"

        return answer, docs, top_funds

    # 3 — Generic RAG fallback
    combined = "\n".join([d["text"] for d in docs])
    return combined, docs, []

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    retrieval: str = "semantic"   # ADDED

class QueryResponse(BaseModel):
    answer: str
    sources: list
    top_funds: list | None = None   # ADDED

def cast_to_python(val):
    if isinstance(val, (np.generic, np.float64, np.int64)):
        return val.item()
    return val

@app.post("/query", response_model=QueryResponse)
def query_api(request: QueryRequest):
    answer, sources, top_funds = enhanced_rag_pipeline(request.query, request.retrieval)

    clean_sources = [{k: cast_to_python(v) for k, v in s.items()} for s in sources]

    return {
        "answer": answer,
        "sources": clean_sources,
        "top_funds": top_funds  # NEW structured output
    }
