import pandas as pd
import plotly.express as px
import streamlit as st

import config
from subsense_engine import (
    ModClassifier,
    RAGEngine,
    RedditFetcher,
    TrendEngine,
    ViralityPredictor,
    run_pipeline,
    summarize,
)

st.set_page_config(page_title="SubSense AI", page_icon="🧠", layout="wide")

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

st.title("🧠 SubSense: GenAI Reddit Intelligence")
st.markdown("### *Beyond Social Listening — True Community Understanding*")


# --- 1. Sidebar Configuration ---
with st.sidebar:
    st.header("🔧 Configuration")

    reddit_ready = bool(config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET)
    if reddit_ready:
        st.success("✅ Reddit credentials loaded.")
    else:
        st.error(
            "❌ Reddit credentials missing. Reddit returns 403 for anonymous access. "
            "Create a free *script* app at reddit.com/prefs/apps and set "
            "`REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` in `.env`."
        )

    # Key handling: environment first, manual entry as a fallback.
    api_key = st.text_input("Gemini API Key", value=config.GEMINI_API_KEY, type="password",
                            help="Only needed for the chat tab. Leave blank to run analysis only.")
    if not api_key:
        st.info("💬 Chat is disabled without a Gemini key. Everything else still works.")

    st.markdown("---")
    st.subheader("🎯 Target Communities")
    sub_input = st.text_input("Subreddits (comma separated)", "AI_India, MachineLearning, startups")

    st.subheader("⏳ Time & Scope")
    time_filter = st.selectbox("Time Range", list(config.TIME_FILTERS),
                               index=list(config.TIME_FILTERS).index("month"))
    post_limit = st.slider("Max Posts per Sub", 50, 500, 100)

    analyze_btn = st.button("🚀 Launch Analysis", disabled=not reddit_ready)

    st.markdown("---")
    st.info("💡 **Pro Tip**: Use the 'all' time range for deep historical training.")


# --- 2. Initialize Engines ---
@st.cache_resource
def load_offline_engines():
    """Layers that need no credentials, so the dashboard works without any key."""
    return TrendEngine(), ViralityPredictor(), ModClassifier()


@st.cache_resource
def load_rag(key: str):
    """Returns (engine, error). Kept separate so a bad key cannot break the dashboard."""
    if not key:
        return None, "No Gemini API key provided."
    try:
        return RAGEngine(key), None
    except Exception as e:
        return None, str(e)


trend_engine, predictor, mod_classifier = load_offline_engines()
rag, rag_error = load_rag(api_key)


@st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
def get_reddit_data(subreddits_list, time_filter, limit):
    """Cached fetch. Returns (dataframe, error) so failures survive the cache."""
    fetcher = RedditFetcher()
    df = fetcher.fetch_data(subreddits_list, time_filter, limit)
    return df, fetcher.last_error


# --- 3. Main Application Logic ---
if analyze_btn:
    subs = [s.strip() for s in sub_input.split(',') if s.strip()]
    if not subs:
        st.error("❌ Enter at least one subreddit.")
    else:
        with st.spinner("📡 Fetching & Normalizing Data (Cached)..."):
            try:
                df, fetch_error = get_reddit_data(subs, time_filter, post_limit)
            except Exception as e:
                st.error(f"Data Fetch Failed: {e}")
                st.stop()

        if not df.empty:
            st.session_state['raw_df'] = df
            st.session_state.pop('df', None)
            st.success(f"✅ Loaded {len(df)} posts from {len(subs)} communities.")
        else:
            st.error(f"❌ {fetch_error or 'No data found. Check the subreddit names.'}")


# Process once per fetch rather than on every widget interaction.
if 'raw_df' in st.session_state and 'df' not in st.session_state:
    with st.spinner("🧠 Running Intelligence Layers..."):
        try:
            st.session_state['df'] = run_pipeline(
                st.session_state['raw_df'],
                trend_engine=trend_engine,
                predictor=predictor,
                mod_classifier=mod_classifier,
            )
        except Exception as e:
            st.error(f"Intelligence Layer Error: {e}")

if 'df' in st.session_state:
    df = st.session_state['df']
    stats = summarize(df)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧠 Deep Insights", "💬 Ask SubSense", "⚙️ MLOps"])

    # TAB 1: Dashboard
    with tab1:
        st.subheader("Bi-Directional Market Pulse")
        try:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Volume", stats['total_posts'])
            col2.metric("Avg Engagement", stats['avg_score'])
            col3.metric("Video/Image Content", stats['media_posts'])
            col4.metric("High Risk Posts", stats['high_risk_posts'])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔥 Trending Topics")
                topics = TrendEngine.top_topics(df)
                if topics.empty:
                    st.info("No topics extracted.")
                else:
                    fig_topics = px.bar(topics, x='posts', y='topic_keywords', orientation='h',
                                        color='posts', color_continuous_scale='Viridis',
                                        labels={'posts': 'Posts', 'topic_keywords': 'Topic'})
                    fig_topics.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_topics, use_container_width=True)

            with c2:
                st.markdown("#### ❤️ Sentiment Distribution")
                if 'sentiment' in df.columns:
                    fig_sent = px.histogram(df, x='sentiment', nbins=20,
                                            color_discrete_sequence=['#FF4B4B'])
                    st.plotly_chart(fig_sent, use_container_width=True)
                else:
                    st.warning("Sentiment data unavailable.")

            st.markdown("#### 🕒 When This Community Posts")
            hourly = (pd.to_datetime(df['created_utc'], errors='coerce').dt.hour
                        .value_counts().sort_index().rename_axis('hour').reset_index(name='posts'))
            if not hourly.empty:
                st.plotly_chart(
                    px.bar(hourly, x='hour', y='posts', color_discrete_sequence=['#FF914D'],
                           labels={'hour': 'Hour (UTC)', 'posts': 'Posts'}),
                    use_container_width=True,
                )

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
            if risky_posts['mod_risk_score'].max() == 0:
                st.success("No posts show mod-risk signals in this sample.")
            for _, post in risky_posts.iterrows():
                if post['mod_risk_score'] == 0:
                    continue
                st.markdown(f"**{post['title']}** (Risk: {post['mod_risk_score']:.1f})")
                st.caption(
                    f"r/{post['subreddit']} • Ratio {post['upvote_ratio']:.2f} • "
                    f"Signals: {post['risk_reasons']}"
                )
                st.divider()

        with c2:
            st.subheader("🚀 Virality Sandbox")
            st.markdown("Test your titles against the **trained XGBoost model**.")

            test_title = st.text_input("Draft Title", "How to build a SaaS in 2 weeks")
            test_media = st.selectbox("Media Type", ["text", "image", "video", "gallery", "link"])
            test_hour = st.slider("Posting Hour (UTC)", 0, 23, 14)
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            test_day = st.selectbox("Posting Day", days)

            if st.button("Predict Potential"):
                if predictor.is_trained:
                    score = predictor.predict_new(test_title, test_media, test_hour,
                                                  day_of_week=days.index(test_day))
                    st.balloons()
                    st.success(f"🔮 Predicted Score: **{int(score)}**")
                else:
                    st.warning("⚠️ Model not trained yet. Train it in the MLOps tab.")

    # TAB 3: RAG Chat
    with tab3:
        st.subheader("💬 Chat with the Data")
        if not rag:
            st.error(f"RAG Engine unavailable: {rag_error}")
        else:
            if not rag.is_indexed():
                st.info("Build the knowledge base in the ⚙️ MLOps tab before asking questions.")

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
                            if context:
                                with st.expander("📚 View Sources"):
                                    st.markdown(context)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response})
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
            st.caption(f"Features: {', '.join(ViralityPredictor.FEATURES)}")

            if st.button("🔄 Force Retrain (Current Data)"):
                with st.spinner("Training & Saving Model..."):
                    st.session_state['df'] = predictor.train_and_score(df, force_retrain=True)
                if predictor.is_trained:
                    st.success("Model retrained and saved.")
                else:
                    st.warning("Training needs at least 10 posts. Widen the time range.")

        with c2:
            st.markdown("### 📚 Knowledge Base")
            st.info(f"Status: **{len(rag.docs) if rag else 0} Docs Indexed**")
            st.markdown(f"Path: `{rag.persist_dir if rag else 'N/A'}`")
            st.caption("Indexing calls the Gemini embedding API once per batch of posts.")

            if st.button("🔄 Build / Re-Index Knowledge Base", disabled=rag is None):
                with st.spinner("Indexing & Persisting..."):
                    ok = rag.index_data(df, force_reindex=True)
                if ok:
                    st.success(f"Knowledge base updated on disk ({len(rag.docs)} docs).")
                else:
                    st.error(f"Indexing failed: {rag.last_error}")
else:
    st.markdown(
        "Configure your credentials in the sidebar, pick a few communities and hit "
        "**Launch Analysis**. Prefer a terminal? `python cli.py --subreddits startups --time week`."
    )
