
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from subsense_engine import RedditFetcher, TrendEngine, ViralityPredictor, ModClassifier, RAGEngine

# Page Config
st.set_page_config(page_title="SubSense AI", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fff; }
    h1, h2, h3 { color: #00e5ff !important; font-family: 'Inter', sans-serif; }
    .stButton>button { background: linear-gradient(45deg, #00e5ff, #2979ff); border: none; font-weight: bold; color: black; }
    div[data-testid="stMetric"] { background: #1a1c24; padding: 10px; border-radius: 8px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# Session State for Persistance
if 'data' not in st.session_state:
    st.session_state.data = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'virality_model' not in st.session_state:
    st.session_state.virality_model = None

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluoro/96/artificial-intelligence.png", width=60)
    st.title("SubSense AI")
    st.caption("Reddit Intelligence System")
    
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.subheader("🎯 Target")
    sub_input = st.text_input("Subreddits", "AI_India, MachineLearning, startups")
    
    col1, col2 = st.columns(2)
    with col1:
        time_filter = st.selectbox("Time Range", ["day", "week", "month", "year", "all"], index=2)
    with col2:
        limit = st.slider("Limit", 50, 500, 100, 50)
        
    run_btn = st.button("🚀 Launch Analysis", use_container_width=True)
    
    st.markdown("---")
    st.info("System Modules:\n- 📈 Trend Engine (Unsupervised)\n- 🚨 Mod Classifier (Heuristic)\n- 🚀 Virality ML (XGBoost)\n- 🧠 RAG Knowledge Store")

# --- Main Logic ---
fetcher = RedditFetcher()
trend_engine = TrendEngine()
mod_classifier = ModClassifier()
virality_predictor = ViralityPredictor()

if run_btn:
    if not api_key:
        # Fallback for demo if users keeps forgetting, but ideally warn
        # st.warning("Using limited demo key...") 
        api_key = "AIzaSyBOu7AaLAuFg0JiFzE4Qm8jtL4tTMdUu_o"

    with st.spinner("📡 Fetching & Normalizing Data..."):
        subs = [s.strip() for s in sub_input.split(',')]
        df = fetcher.fetch_data(subs, time_filter, limit)
        
    if not df.empty:
        with st.spinner("⚙️ Running Intelligence Layers (ML + NLP)..."):
            # 1. Trends
            df = trend_engine.extract_trends(df)
            # 2. Mod Risk
            df = mod_classifier.score_risk(df)
            # 3. Virality (Train on this batch for demo)
            df = virality_predictor.train_and_score(df)
            st.session_state.virality_model = virality_predictor
            
            # 4. RAG Indexing
            if api_key:
                rag = RAGEngine(api_key)
                rag.index_data(df)
                st.session_state.rag_engine = rag
            
            st.session_state.data = df
        st.success("Analysis Complete!")
    else:
        st.error("No data found. Check subreddit names.")

# --- Dashboard UI ---
if st.session_state.data is not None:
    df = st.session_state.data
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Trends", "🧠 Core Intelligence", "💬 Ask SubSense"])
    
    # TAB 1: DASHBOARD
    with tab1:
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Posts", len(df))
        m2.metric("Avg Score", int(df['score'].mean()))
        m3.metric("Video/Image Content", f"{len(df[df['media_type']!='text'])}")
        m4.metric("Risk Flags", len(df[df['mod_risk_score'] > 0.5]))
        
        # Charts
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📈 Trending Topics (Clustering)")
            # Aggregation by topic
            topic_counts = df['topic_keywords'].value_counts().reset_index()
            topic_counts.columns = ['Keywords', 'Volume']
            fig = px.bar(topic_counts.head(10), x='Volume', y='Keywords', orientation='h', color='Volume', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("🎭 Sentiment Distribution")
            fig2 = px.histogram(df, x="sentiment", nbins=20, title="Sentiment Polarity")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.subheader("📂 Raw Data Inspector")
        st.dataframe(df[['subreddit', 'title', 'score', 'topic_keywords', 'mod_risk_score', 'sentiment']], use_container_width=True)

    # TAB 2: INTELLIGENCE
    with tab2:
        col_A, col_B = st.columns(2)
        
        with col_A:
            st.subheader("🚨 Moderator Signal Classifier")
            st.caption("Detects posts with high risk of removal or low value.")
            
            risky_posts = df[df['mod_risk_score'] > 0.5].sort_values('mod_risk_score', ascending=False)
            if not risky_posts.empty:
                for _, row in risky_posts.head(5).iterrows():
                    with st.expander(f"⚠️ {row['title'][:50]}... (Risk: {row['mod_risk_score']})"):
                        st.write(f"**Reasons**: Locked={row['is_locked']}, Ratio={row['upvote_ratio']}, Sentiment={row['sentiment']:.2f}")
                        st.write(row['body'][:200])
            else:
                st.success("No high-risk posts detected.")
                
        with col_B:
            st.subheader("🚀 Virality Simulator (ML)")
            st.caption("Predict engagement for a hypothetical post based on current subreddit trends.")
            
            sim_title = st.text_input("Draft Title", "My new AI startup")
            sim_media = st.selectbox("Media Type", ["text", "image", "video"])
            sim_hour = st.slider("Posting Hour (UTC)", 0, 23, 12)
            
            if st.button("Predict Performance"):
                if st.session_state.virality_model:
                    pred_score = st.session_state.virality_model.predict_new(sim_title, sim_media, sim_hour)
                    st.metric("Predicted Score", f"{int(pred_score)} ▲", delta="Based on XGBoost Model")
                else:
                    st.warning("Run analysis first to train the model.")

    # TAB 3: RAG CHAT
    with tab3:
        st.subheader("💬 Chat with these Subreddits")
        st.caption("Ask questions like: 'What are people complaining about?' or 'Summarize the hiring trends'.")
        
        user_q = st.chat_input("Ask SubSense...")
        if user_q:
            if st.session_state.rag_engine:
                with st.chat_message("user"):
                    st.write(user_q)
                    
                with st.spinner("Thinking..."):
                    answer, context = st.session_state.rag_engine.query(user_q)
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("View Sources"):
                        st.text(context)
            else:
                st.error("RAG Engine not initialized. Please run analysis first.")
else:
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h1>🧠 Welcome to SubSense</h1>
        <p>The Advanced GenAI Reddit Intelligence System.</p>
        <p>👈 Start by selecting your target communities in the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)

