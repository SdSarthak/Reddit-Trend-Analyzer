import streamlit as st
import requests
import google.generativeai as genai
import json
import pandas as pd
from typing import List, Dict
import os
from dotenv import load_dotenv
from subsense_engine import RedditFetcher, TrendEngine, ViralityPredictor, ModClassifier, RAGEngine

# Load Environment Variables
load_dotenv()

# Page Config
st.set_page_config(page_title="SubSense AI", page_icon="🧠", layout="wide")

# Custom CSS for "Premium" Feel
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stApp h1 { color: #FF4B4B; font-family: 'Helvetica Neue', sans-serif; }
    div.stButton > button {
        background: linear-gradient(45deg, #FF4B4B, #FF914D);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover { transform: scale(1.05); box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4); }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #363940;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Application Title
st.title("🧠 SubSense: GenAI Reddit Intelligence")
st.markdown("### *Beyond Social Listening — True Community Understanding*")

# --- 1. Sidebar Configuration ---
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key Handling (Env -> Input Fallback)
    env_api_key = os.getenv("GEMINI_API_KEY")
    api_key = st.text_input("Gemini API Key", value=env_api_key if env_api_key else "", type="password")
    
    if not api_key:
        st.warning("⚠️ Please provide an API Key in .env or here.")
        
    st.markdown("---")
    st.subheader("🎯 Target Communities")
    sub_input = st.text_input("Subreddits (comma separated)", "AI_India, MachineLearning, startups")
    
    st.subheader("⏳ Time & Scope")
    time_filter = st.selectbox("Time Range", ["day", "week", "month", "year", "all"], index=2)
    post_limit = st.slider("Max Posts per Sub", 50, 500, 100) # Increased range for "Production" feel
    
    analyze_btn = st.button("🚀 Launch Analysis")
    
    st.markdown("---")
    st.info("💡 **Pro Tip**: Use 'all' time range for deep historical training.")

# --- 2. Initialize Engines (Cached Resource) ---
@st.cache_resource
def load_engines(api_key):
    # Only load RAG if we have a key
    rag = RAGEngine(api_key) if api_key else None
    return RedditFetcher(), TrendEngine(), ViralityPredictor(), ModClassifier(), rag

fetcher, trend_engine, predictor, mod_classifier, rag = load_engines(api_key)

# --- 3. Cached Data Fetching ---
@st.cache_data(ttl=3600, show_spinner=False) # Cache for 1 hour
def get_reddit_data(subreddits_list, time_filter, limit):
    return fetcher.fetch_data(subreddits_list, time_filter, limit)

# --- 4. Main Application Logic ---
if analyze_btn and api_key:
    subs = [s.strip() for s in sub_input.split(',')]
    
    with st.spinner("📡 Fetching & Normalizing Data (Cached)..."):
        try:
            df = get_reddit_data(subs, time_filter, limit=post_limit)
        except Exception as e:
            st.error(f"Data Fetch Failed: {e}")
            st.stop()
            
    if not df.empty:
        # Save to session state to persist across reruns
        st.session_state['df'] = df
        st.success(f"✅ Loaded {len(df)} posts from {len(subs)} communities.")
    else:
        st.error("❌ No data found. Check subreddit names.")

# Check if data exists in session
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Process Data (Apply Intelligence Layers)
    with st.spinner("🧠 Running Intelligence Layers..."):
        try:
            # 1. Trends
            df = trend_engine.extract_trends(df)
            # 2. Mod Risk
            df = mod_classifier.score_risk(df)
            # 3. Virality (Predict Only if trained, or Train if requested in MLOps)
            df = predictor.train_and_score(df) # Logic handles persistence
            # 4. RAG Indexing (Only if not already indexed? For now re-index active session data)
            if rag: rag.index_data(df)
        except Exception as e:
            st.error(f"Intelligence Layer Error: {e}")

    # --- UI TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧠 Deep Insights", "💬 Ask SubSense", "⚙️ MLOps"])
    
    # TAB 1: Dashboard
    with tab1:
        st.subheader("Bi-Directional Market Pulse")
        try:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Volume", f"{len(df)}")
            col2.metric("Avg Engagement", f"{int(df['score'].mean())}")
            col3.metric("Video/Image Content", f"{len(df[df['media_type'] != 'text'])}")
            col4.metric("High Risk Posts", f"{len(df[df['mod_risk_score'] > 1.0])}")
            
            # Visuals
            import plotly.express as px
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔥 Trending Topics")
                topic_counts = df['topic_keywords'].value_counts().reset_index()
                topic_counts.columns = ['Topic', 'Count']
                fig_topics = px.bar(topic_counts, x='Count', y='Topic', orientation='h', color='Count', color_continuous_scale='Viridis')
                st.plotly_chart(fig_topics, use_container_width=True)
                
            with c2:
                st.markdown("#### ❤️ Sentiment Distribution")
                if 'sentiment' in df.columns:
                    fig_sent = px.histogram(df, x='sentiment', nbins=20, color_discrete_sequence=['#FF4B4B'])
                    st.plotly_chart(fig_sent, use_container_width=True)
                else:
                    st.warning("Sentiment data unavailable.")

            with st.expander("🔎 Inspect Raw Data"):
                st.dataframe(df)
        except Exception as e:
            st.error(f"Dashboard Rendering Error: {e}")

    # TAB 2: Insights
    with tab2:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("🚨 Mod Queue Simulator")
            risky_posts = df.sort_values('mod_risk_score', ascending=False).head(5)
            for _, post in risky_posts.iterrows():
                with st.container():
                    st.markdown(f"**{post['title']}** (Risk: {post['mod_risk_score']:.1f})")
                    st.caption(f"Reason: Low Ratio ({post['upvote_ratio']}) • Locked: {post['is_locked']}")
                    st.divider()

        with c2:
            st.subheader("🚀 Virality Sandbox")
            st.markdown("Test your titles against our **trained XGBoost model**.")
            
            test_title = st.text_input("Draft Title", "How to build a SaaS in 2 weeks")
            test_media = st.selectbox("Media Type", ["text", "image", "video"])
            test_hour = st.slider("Posting Hour (UTC)", 0, 23, 14)
            
            if st.button("Predict Potential"):
                if predictor.is_trained:
                    score = predictor.predict_new(test_title, test_media, test_hour)
                    st.balloons()
                    st.success(f"🔮 Predicted Score: **{int(score)}**")
                else:
                    st.warning("⚠️ Model not trained yet. Please train in MLOps tab.")

    # TAB 3: RAG Chat
    with tab3:
        st.subheader("💬 Chat with the Data")
        if not rag:
            st.error("RAG Engine unavailable (Missing Key or Libs).")
        else:
            # Simple Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask about trends, complaints, or insights..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("🤖 Thinking..."):
                        try:
                            response, context = rag.query(prompt)
                            st.markdown(response)
                            with st.expander("📚 View Sources"):
                                st.markdown(context)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Chat Error: {e}")

    # TAB 4: MLOps
    with tab4:
        st.subheader("⚙️ System Operations")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🧠 Virality Model")
            st.info(f"Status: **{'Trained & Loaded' if predictor.is_trained else 'Not Trained'}**")
            st.markdown(f"Path: `{predictor.model_path}`")
            
            if st.button("🔄 Force Retrain (Current Data)"):
                with st.spinner("Training & Saving Model..."):
                    predictor.train_and_score(df, force_retrain=True)
                st.success("Model Retrained & Saved!")
                
        with c2:
            st.markdown("### 📚 Knowledge Base")
            st.info(f"Status: **{len(rag.docs) if rag else 0} Docs Indexed**")
            st.markdown(f"Path: `{rag.persist_dir if rag else 'N/A'}`")
            
            if st.button("🔄 Re-Index Knowledge Base"):
                if rag:
                    with st.spinner("Re-indexing & Persisting..."):
                        rag.index_data(df, force_reindex=True)
                    st.success("Knowledge Base Updated on Disk!")
