# Mini RAG with Financial Data Context

This application is a **mini Retrieval-Augmented Generation (RAG) system** designed to answer financial queries related to mutual funds. It integrates both **textual FAQ knowledge** and **quantitative fund performance data** to provide:

* Direct definitions for financial terms
* Analytical comparisons of mutual funds (e.g., best CAGR, lowest volatility)
* Hybrid RAG responses combining retrieved context and structured ranking outputs

The system supports **semantic**, **lexical**, and **hybrid** retrieval modes and uses intent detection, metric extraction, and deterministic ranking to produce accurate, structured answers. All embeddings and indexes are cached for near-instant startup.

# Instructions to Run Locally
Below are the complete steps to run the project locally:

### **1. Clone the Repository**
```bash
git clone <your-github-repo-url>
cd <your-repo-folder>
```

### **2. Create and Activate a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Application**
```bash
uvicorn main:app --reload
```
Your app should now be running locally at:
http://127.0.0.1:8000

### **5. Access the API Documentation**
Open your browser and navigate to:
http://127.0.0.1:8000/docs

This will open the **Swagger UI**, where you can interact with the API.

### **6. Test the /query Endpoint**
In the Swagger UI, you can test analytical queries such as:
```json
{
  "query": "Which funds have the best Sharpe ratio in the last 3 years?",
  "retrieval": "semantic"
}
```
In the Swagger UI:
1. Scroll to the **POST /query** section.
2. Click **Try it out**.
3. Enter your query text into the input box (e.g., "What is the Sharpe ratio?", "Compare funds by volatility", etc.).
4. Click **Execute** to see the JSON response.

---

# Design

## 1. System Architecture Overview
The system is structured into three major functional layers:

### **A. Retrieval Layer (Deterministic + Cached)**
All heavy computational artifacts are cached to ensure ~0.1s startup time:
* **SentenceTransformer embeddings** for semantic search
* **FAISS index** for fast vector search
* **TF-IDF matrix** for lexical search

This makes the retrieval pipelines deterministic and extremely fast after initial caching.

### **B. Intent Understanding Layer (Intelligent)**
Intent classification is performed using semantic similarity between user queries and predefined intent templates. This enables the system to differentiate:
* Definition queries
* Comparison (analytic) queries
* Generic descriptive queries

### **C. Analytical Layer (Deterministic)**
For comparison queries, the system:
* Detects the metric semantically
* Detects ranking direction via rules + fallback semantic similarity
* Ranks funds deterministically using Pandas sorting
* Produces Top‑K structured results

This ensures stable, interpretable analytical outputs.

## 2. Control Flow Overview
1. User sends query with retrieval mode selected (semantic / lexical / hybrid)
2. Relevant context retrieved
3. Intent detected
4. Metric + direction detected (if applicable)
5. Funds ranked deterministically
6. Final structured response returned

![System Flow](https://github.com/KayalvizhiT513/Mini-RAG-with-Financial-Data-Context/blob/main/flow.png)

---

# Trade-offs

### **1. Semantic Retrieval Limitations**
The embedding model sometimes associates conceptually opposite terms due to contextual proximity. Example:
* "stability" is semantically close to "volatility", but the meanings conflict.
* Query: *"Recommend stable funds"* may return funds with **high volatility**, because embeddings pick similarity, not polarity.

### **2. Lexical vs Semantic Behavior Mismatch**
Lexical search precisely matches FAQ wording. Semantic search may:
* Over-prioritize fund descriptions
* Hallucinate relevance when FAQ answers contain fewer embedding matches

Example:
* Query: *"What is the Sharpe ratio"*
  * Lexical → correct FAQ returned
  * Semantic → fund descriptions dominate due to dense numeric/financial text
  * Hybrid → correct result (balances both)

---

# Assumptions

### **1. Metrics Are Precomputed**
All fund metrics (CAGR, Sharpe ratio, volatility) exist in the dataset and are assumed accurate.

### **2. Embeddings Represent Funds Well Enough**
The model-generated embeddings are assumed to meaningfully represent:
* Fund characteristics
* FAQs
* Metric-related phrases

### **3. Metric Keywords Are Treated as Positively Correlated**
Words related to a metric (e.g., "stability", "risk", "volatility") are assumed to refer to the metric itself, not its polarity.

