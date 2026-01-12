import requests
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Essential ML imports - fail hard if missing because they are core features now
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    logger.error(f"Critical Import Error: {e}. Please ensure sentence-transformers and faiss-cpu are installed.")
    # Define dummy for safety if user runs without satisfying requirements immediately
    class SentenceTransformer:
        def __init__(self, model_name): pass
        def encode(self, texts): return np.array([])
    faiss = None

import google.generativeai as genai

# --- 1. Data Acquisition & Normalization ---

class RedditFetcher:
    """Handles fetching data locally via JSON endpoints (No PRAW needed for basic demo)."""
    
    def __init__(self):
        self.headers = {'User-Agent': 'SubSense/2.0'}
        logger.info("RedditFetcher initialized.")

    def fetch_data(self, subreddits: List[str], time_filter: str = 'month', limit: int = 100) -> pd.DataFrame:
        all_data = []
        logger.info(f"Fetching data for subreddits: {subreddits}, Time: {time_filter}, Limit: {limit}")
        
        # map UI time filter to Reddit params
        # time_filter options: 'day', 'week', 'month', 'year', 'all'
        
        for sub in subreddits:
            sub = sub.replace('r/', '').strip()
            # Fetch 'top' posts to get best data for time range
            url = f"https://www.reddit.com/r/{sub}/top/.json?t={time_filter}&limit={limit}"
            
            try:
                # We need pagination loop here similar to before
                results = self._fetch_paginated(url, limit)
                for post in results:
                    post['source_subreddit'] = sub
                all_data.extend(results)
                logger.info(f"Successfully fetched {len(results)} posts from r/{sub}")
            except Exception as e:
                logger.error(f"Error fetching r/{sub}: {e}")
                
        df = self._normalize(all_data)
        logger.info(f"Total normalized records: {len(df)}")
        return df

    def _fetch_paginated(self, base_url: str, limit: int):
        children = []
        after = None
        
        while len(children) < limit:
            req_url = f"{base_url}&after={after}" if after else base_url
            try:
                resp = requests.get(req_url, headers=self.headers)
                if resp.status_code != 200: 
                    logger.warning(f"Failed request to {req_url}: Score {resp.status_code}")
                    break
                
                data = resp.json()
                new_children = [c['data'] for c in data['data']['children']]
                if not new_children: break
                
                children.extend(new_children)
                after = data['data'].get('after')
                if not after: break
                
            except Exception as e:
                logger.error(f"Pagination error: {e}")
                break
            
            # Simple rate limit prevention
            # time.sleep(0.1) 
            
        return children[:limit]

    def _normalize(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Normalized Universal Schema."""
        normalized = []
        for p in raw_data:
            # Media type detection
            media_type = 'text'
            if p.get('is_video'): media_type = 'video'
            elif p.get('url', '').endswith(('.jpg', '.png', '.gif')): media_type = 'image'
            elif 'gallery_data' in p: media_type = 'gallery'

            normalized.append({
                "post_id": p.get('id'),
                "subreddit": p.get('source_subreddit', p.get('subreddit')),
                "title": p.get('title', ''),
                "body": p.get('selftext', ''),
                "author": p.get('author', ''),
                "created_utc": datetime.fromtimestamp(p.get('created_utc', 0)),
                "flair": p.get('link_flair_text', ''),
                "ups": p.get('ups', 0),
                "score": p.get('score', 0),
                "num_comments": p.get('num_comments', 0),
                "upvote_ratio": p.get('upvote_ratio', 0.0),
                "is_text": p.get('is_self', True),
                "media_type": media_type,
                "is_stickied": p.get('stickied', False),
                "is_locked": p.get('locked', False),
                "is_nsfw": p.get('over_18', False),
                "url": p.get('url', '')
            })
        return pd.DataFrame(normalized)

# --- 2. Intelligence Layers ---

class TrendEngine:
    """Unsupervised ML for Topic Modeling."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.kmeans = KMeans(n_clusters=5, random_state=42) # Default 5 clusters for demo
        logger.info("TrendEngine initialized.")

    def extract_trends(self, df: pd.DataFrame):
        if df.empty: 
            logger.warning("Empty DataFrame passed to TrendEngine.")
            return df
        
        logger.info("Extracting trends from dataset.")
        # Combine title and body for text
        text_data = df['title'] + " " + df['body'].fillna('')
        
        # Vectorize
        try:
            tfidf_matrix = self.vectorizer.fit_transform(text_data)
            
            # Cluster
            num_clusters = min(5, len(df)//5) # Adaptive clusters
            if num_clusters < 2: num_clusters = 1
            
            self.kmeans = KMeans(n_clusters=num_clusters)
            df['topic_cluster'] = self.kmeans.fit_predict(tfidf_matrix)
            
            # Extract keywords for each cluster
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_names = {}
            
            for i in range(num_clusters):
                center = self.kmeans.cluster_centers_[i]
                top_ind = center.argsort()[:-6:-1] # Top 5 words
                keywords = [feature_names[ind] for ind in top_ind]
                cluster_names[i] = ", ".join(keywords)
                
            df['topic_keywords'] = df['topic_cluster'].map(cluster_names)
            logger.info("Trend extraction complete.")
        except Exception as e:
            logger.error(f"Trend extraction failed: {e}")
            df['topic_keywords'] = "General"
            
        return df

class ViralityPredictor:
    """Predicts engagement (Regression)."""
    
    def __init__(self):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
        logger.info("ViralityPredictor initialized.")
        
    def train_and_score(self, df: pd.DataFrame):
        if len(df) < 10: 
            logger.warning("Not enough data to train ViralityPredictor (needs 10+ samples).")
            return df
        
        logger.info(f"Training Virality model on {len(df)} posts.")
        # Features
        # 1. Title Length
        df['title_len'] = df['title'].apply(len)
        # 2. Is Media
        df['has_media'] = (df['media_type'] != 'text').astype(int)
        # 3. Hour of day
        df['hour'] = df['created_utc'].dt.hour
        
        # X and Y
        X = df[['title_len', 'has_media', 'hour']]
        y = df['score']
        
        try:
            self.model.fit(X, y)
            # Predict "Potential" (just to show simulator capability)
            df['predicted_score'] = self.model.predict(X)
            logger.info("Virality model trained successfully.")
        except Exception as e:
            logger.error(f"Virality model training failed: {e}")
            
        return df
        
    def predict_new(self, title: str, media_type: str, hour: int):
        # Demo prediction
        df = pd.DataFrame([{
            'title_len': len(title), 
            'has_media': 1 if media_type != 'text' else 0, 
            'hour': hour
        }])
        try:
            pred = self.model.predict(df)[0]
            logger.info(f"Predicted score {pred:.2f} for new post: '{title}'")
            return pred
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0

class ModClassifier:
    """Heuristic & NLP based Mod signals."""
    def __init__(self):
        logger.info("ModClassifier initialized.")
    
    def score_risk(self, df: pd.DataFrame):
        logger.info("Calculating Mod Risk Scores.")
        # 1. Heuristic Risk
        # Locked posts are potentially controversial
        df['mod_risk_score'] = 0.0
        
        # If locked but not stickied (usually means deleted/bad behavior)
        df.loc[df['is_locked'] & ~df['is_stickied'], 'mod_risk_score'] += 0.8
        
        # Low upvote ratio (< 0.6)
        df.loc[df['upvote_ratio'] < 0.60, 'mod_risk_score'] += 0.5
        
        # Sentiment Analysis (Negative sentiment might be riskier in some contexts)
        try:
            df['sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            df['sentiment'] = 0.0
        
        return df

class RAGEngine:
    """GenAI RAG Layer."""
    
    def __init__(self, api_key):
        self.index = None
        self.docs = []
        logger.info("Initializing RAGEngine.")
        
        if 'SentenceTransformer' not in globals() or SentenceTransformer.__module__ == __name__:
             # Check if it was dummy
             try:
                 from sentence_transformers import SentenceTransformer
                 self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                 logger.info("SentenceTransformer loaded.")
             except Exception as e:
                 logger.error(f"Failed to load real SentenceTransformer: {e}")
                 self.embedder = None
        else:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def index_data(self, df: pd.DataFrame):
        if not self.embedder:
            logger.error("Cannot index data: Embedder not available.")
            return

        logger.info(f"Indexing {len(df)} documents for RAG.")
        # Prepare docs
        self.docs = df.to_dict('records')
        texts = [f"Sub: {d['subreddit']} | Title: {d['title']} | Body: {d['body'][:500]}" for d in self.docs]
        
        # Embed
        try:
            embeddings = self.embedder.encode(texts)
            
            # FAISS Index
            if faiss:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(embeddings)
                logger.info("FAISS Index build complete.")
            else:
                logger.error("FAISS not available.")
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
        
    def query(self, user_query: str):
        if not self.index: return "Index not built.", ""
        
        logger.info(f"Processing RAG Query: {user_query}")
        # Search
        q_embed = self.embedder.encode([user_query])
        D, I = self.index.search(q_embed, k=5) # Top 5 relevant posts
        
        # Retrieve
        context = ""
        for idx in I[0]:
            if idx < len(self.docs):
                d = self.docs[idx]
                context += f"- [r/{d['subreddit']}] {d['title']} (Score: {d['score']})\n"
                
        # Generate
        prompt = f"""
        Context from Subreddit Analysis:
        {context}
        
        User Query: {user_query}
        
        Answer based on the context provided. Cite specific posts if relevant.
        """
        try:
            resp = self.model.generate_content(prompt)
            logger.info("RAG Response generated.")
            return resp.text, context
        except Exception as e:
            logger.error(f"GenAI generation failed: {e}")
            return "Error generating response.", context
