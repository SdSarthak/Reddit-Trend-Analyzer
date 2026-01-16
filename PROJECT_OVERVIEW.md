# 🧠 SubSense: GenAI Reddit Intelligence System

**SubSense** is a portfolio-tier generative AI application designed to transform raw Reddit discussions into actionable market intelligence. Unlike static keyword scrapers, SubSense uses a multi-layered, **production-ready** AI pipeline to understand, analyze, and interact with community data in real-time.

---

## 🚀 Product Vision

### The Problem

Traditional social listening tools are:

- **Static**: Relying on simple keyword counting.
- **Transient**: Insights are lost once the session ends.
- **Shallow**: Unable to parse context, sarcasm, or "virality" factors.

### The SubSense Solution

SubSense is a **Dynamic Intelligence Engine**. It allows users to plug in _any_ combination of subreddits (e.g., `r/startups` + `r/AI_India`) and instantly generates a strategic dashboard. It doesn't just read posts; it **predicts trends**, **assesses risk**, and **answers questions**, while **learning and remembering** from historical data.

---

## ⚙️ Core Intelligence Layers

The system is built on a modular "Ladder of Intelligence":

### 1. 📡 Universal Data Fetcher (Cached)

- **Smart Caching**: Implements TTL-based caching to prevent redundant API calls and respect rate limits.
- **Universal Normalization**: Converts chaotic Reddit JSON into a clean, ML-ready Parquet schema.

### 2. 📈 Trend Engine (Unsupervised ML)

- **Algorithm**: TF-IDF Vectorization + K-Means Clustering.
- **Goal**: Automatically groups hundreds of posts into coherent "Topics" (e.g., "Hiring Trends", "LLM Fatigue").

### 3. 🚨 Mod & Risk Classifier (Heuristic Engine)

- **Algorithm**: Rule-based Heuristics + Sentiment Analysis (TextBlob).
- **Goal**: Flags posts likely to be removed. Detects "Locked" threads, low upvote ratios (<60%), and negative sentiment spikes.

### 4. 🚀 Virality Simulator (Persistent ML)

- **Algorithm**: XGBoost Regressor (Persisted to Disk).
- **MLOps**: Supports **Model Persistence**. The model can be trained on historical data, saved (`virality_model.json`), and loaded for instant inference in future sessions.
- **Goal**: Predicts the "Potential Score" of a _hypothetical_ post based on its title length, media type, and posting hour.

### 5. 🧠 RAG Knowledge Store (GenAI + Vector DB)

- **Algorithm**: Gemini (`text-embedding-004`) + FAISS / Numpy + Gemini 2.5 Flash.
- **Persistence**: Embeddings and metadata are serialized to disk (`rag_store/`), enabling a "Long-Term Memory" for the AI.
- **Smart Indexing**: Uses rate-limited batching to handle large datasets without hitting API quotas.
- **Goal**: Enables a "Chat with Data" experience with citation-backed answers.

---

## 🏭 Production Engineering (MLOps)

SubSense implements core MLOps principles:

- **Model Registry**: ML models are serialized and versioned locally, decoupling "Training" from "Inference".
- **State Management**: `joblib` and filesystem persistence ensure the AI doesn't "forget" when the server restarts.
- **Robust Error Boundaries**: UI components are wrapped in try/catch blocks to ensure graceful degradation instead of crashing.
- **Environment Config**: Sensitive keys are managed via `.env` files, following 12-Factor App methodology.

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Python) - Custom "Dark Mode" UI.
- **LLM & AI**: Google Gemini 2.5 Flash (Reasoning), Gemini Text Embedding 004.
- **Machine Learning**: `scikit-learn` (Clustering), `xgboost` (Regression), `TextBlob` (NLP).
- **Vector Database**: `FAISS` (Facebook AI Similarity Search) with `Numpy` fallback.
- **Infrastructure**: `python-dotenv` (Config), `joblib` (Serialization).

---

## 🖥️ Application Architecture

```mermaid
graph TD
    User[User Input] -->|Subreddits + Config| Cache[Caching Layer]
    Cache --> Fetcher[Reddit Fetcher]
    Fetcher -->|Raw JSON| Normalizer[Data Normalizer]
    Normalizer -->|Structured DF| ML_Layer[Intelligence Core]

    subgraph Intelligence Core
        ML_Layer --> Trend[Trend Engine (K-Means)]
        ML_Layer --> Risk[Mod Classifier (Heuristic)]
        ML_Layer --> Viral[Virality Model (XGBoost)]
    end

    Viral <-->|Save/Load| DiskModel[(virality_model.json)]

    ML_Layer --> Vector[RAG Indexer (Gemini 004)]
    Vector <-->|Save/Load| DiskIndex[(rag_store/)]

    Trend --> UI[Streamlit Dashboard]
    Risk --> UI
    Viral --> UI
    Vector -->|Context| Chat[Chat Interface]
    Chat -->|Prompt| LLM[Gemini 2.5]
    LLM --> UI
```

## 📖 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```
3.  **Launch App**:
    ```bash
    streamlit run app.py
    ```
4.  **Operations**:
    - Navigate to the **⚙️ MLOps** tab to Train models or Re-index the Knowledge Base.

---

**Built by [Your Name/Team]**
_Targeting: Market Researchers, DevRel Teams, and Content Creators._
