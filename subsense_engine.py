import time
import os
import joblib
import json
import logging
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import List, Dict, Optional, Any
from datetime import datetime
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Essential ML imports
ML_AVAILABLE = False
try:
    import faiss
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"FAISS not available ({e}). Switching to Numpy for Vector Search.")
    faiss = None

import google.generativeai as genai

# ... [RedditFetcher unchanged] ...
class RedditFetcher:
    """Handles fetching data locally via JSON endpoints (No PRAW needed for basic demo)."""
    
    def __init__(self):
        self.headers = {'User-Agent': 'SubSense/2.0'}
        logger.info("RedditFetcher initialized.")

    def fetch_data(self, subreddits: List[str], time_filter: str = 'month', limit: int = 100) -> pd.DataFrame:
        all_data = []
        logger.info(f"Fetching data for subreddits: {subreddits}, Time: {time_filter}, Limit: {limit}")
        
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


# ... [TrendEngine unchanged] ...
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
        text_data = df['title'] + " " + df['body'].fillna('')
        try:
            tfidf_matrix = self.vectorizer.fit_transform(text_data)
            num_clusters = min(5, len(df)//5) 
            if num_clusters < 2: num_clusters = 1
            self.kmeans = KMeans(n_clusters=num_clusters)
            df['topic_cluster'] = self.kmeans.fit_predict(tfidf_matrix)
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_names = {}
            for i in range(num_clusters):
                center = self.kmeans.cluster_centers_[i]
                top_ind = center.argsort()[:-6:-1] 
                keywords = [feature_names[ind] for ind in top_ind]
                cluster_names[i] = ", ".join(keywords)
            df['topic_keywords'] = df['topic_cluster'].map(cluster_names)
            logger.info("Trend extraction complete.")
        except Exception as e:
            logger.error(f"Trend extraction failed: {e}")
            df['topic_keywords'] = "General"
        return df

class ViralityPredictor:
    """Predicts engagement (Regression) with Persistence."""
    
    def __init__(self, model_path="virality_model.json"):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
        self.model_path = model_path
        self.is_trained = False
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model.load_model(self.model_path)
                self.is_trained = True
                logger.info(f"Loaded Virality Model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load virality model: {e}")
    
    def save_model(self):
        try:
            self.model.save_model(self.model_path)
            logger.info(f"Saved Virality Model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save virality model: {e}")

    def train_and_score(self, df: pd.DataFrame, force_retrain=False):
        if len(df) < 10: 
            logger.warning("Not enough data to train ViralityPredictor.")
            return df
        
        # Prepare Features
        df['title_len'] = df['title'].apply(len)
        df['has_media'] = (df['media_type'] != 'text').astype(int)
        df['hour'] = df['created_utc'].dt.hour
        X = df[['title_len', 'has_media', 'hour']]
        y = df['score']
        
        # If we have a trained model and don't force retrain, JUST predict
        if self.is_trained and not force_retrain:
            try:
                df['predicted_score'] = self.model.predict(X)
                logger.info("Used persisted model for scoring.")
                return df
            except:
                logger.warning("Persisted model failed to predict. Retraining...")
        
        # Train
        try:
            logger.info(f"Training Virality model on {len(df)} posts.")
            self.model.fit(X, y)
            self.is_trained = True
            df['predicted_score'] = self.model.predict(X)
            self.save_model() # Auto-save on successful train
        except Exception as e:
            logger.error(f"Virality model training failed: {e}")
            
        return df
        
    def predict_new(self, title: str, media_type: str, hour: int):
        if not self.is_trained: return 0
        df = pd.DataFrame([{
            'title_len': len(title), 
            'has_media': 1 if media_type != 'text' else 0, 
            'hour': hour
        }])
        try: return self.model.predict(df)[0]
        except: return 0

# ... [ModClassifier unchanged] ...
class ModClassifier:
    """Heuristic & NLP based Mod signals."""
    def __init__(self):
        logger.info("ModClassifier initialized.")
    
    def score_risk(self, df: pd.DataFrame):
        logger.info("Calculating Mod Risk Scores.")
        df['mod_risk_score'] = 0.0
        df.loc[df['is_locked'] & ~df['is_stickied'], 'mod_risk_score'] += 0.8
        df.loc[df['upvote_ratio'] < 0.60, 'mod_risk_score'] += 0.5
        try:
            df['sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            df['sentiment'] = 0.0
        return df

class RAGEngine:
    """GenAI RAG Layer using Gemini Embeddings + FAISS + Persistence."""
    
    def __init__(self, api_key, persist_dir="rag_store"):
        self.index = None
        self.docs = []
        self.doc_embeddings = None
        self.persist_dir = persist_dir
        
        # Setup Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.embedding_model = "models/text-embedding-004"
        
        # Try Loading Existing Index
        self._load_index()

    def _load_index(self):
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir, exist_ok=True)
            return

        index_file = os.path.join(self.persist_dir, "index.faiss")
        docs_file = os.path.join(self.persist_dir, "docs.json")
        embed_file = os.path.join(self.persist_dir, "embeddings.npy")
        
        try:
            if os.path.exists(docs_file):
                with open(docs_file, 'r') as f:
                    self.docs = json.load(f)
                logger.info(f"Loaded {len(self.docs)} docs from disk.")
            
            if faiss and os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
                logger.info("Loaded FAISS index from disk.")
            elif os.path.exists(embed_file):
                self.doc_embeddings = np.load(embed_file)
                logger.info("Loaded Numpy embeddings from disk.")
                
        except Exception as e:
            logger.error(f"Failed to load persisted RAG index: {e}")

    def save_index(self):
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir, exist_ok=True)
            
        try:
            # Save Docs
            with open(os.path.join(self.persist_dir, "docs.json"), 'w') as f:
                json.dump(self.docs, f)
            
            # Save Index or Embeddings
            if self.index and faiss:
                faiss.write_index(self.index, os.path.join(self.persist_dir, "index.faiss"))
            elif self.doc_embeddings is not None:
                np.save(os.path.join(self.persist_dir, "embeddings.npy"), self.doc_embeddings)
                
            logger.info("Successfully persisted RAG index.")
        except Exception as e:
            logger.error(f"Failed to save RAG index: {e}")

    def index_data(self, df: pd.DataFrame, force_reindex=False):
        # If we have docs and don't force reindex, assume we append or just keep current? 
        # For simplicity in this demo: Re-index always replaces current active set if called explicitly, 
        # BUT we could optimize to append. 
        # Given user wants "Train on historical", let's assume we are building a NEW index from this data.
        
        logger.info(f"Indexing {len(df)} documents for RAG.")
        self.docs = df.to_dict('records') # Update Docs
        
        texts = [f"Sub: {d['subreddit']} | Title: {d['title']} | Body: {d['body'][:500]}" for d in self.docs]
        all_embeddings = []
        batch_size = 20
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=batch,
                    task_type="retrieval_document"
                )
                if 'embedding' in result: 
                    all_embeddings.extend(result['embedding'])
                logger.info(f"Embedded batch {i//batch_size + 1}")
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Gemini Embedding batch error: {e}")
                return # Stop if API fails
                
        if not all_embeddings: return

        # Build Index
        try:
            matrix = np.array(all_embeddings).astype('float32')
            self.doc_embeddings = matrix
            
            if faiss:
                self.index = faiss.IndexFlatL2(matrix.shape[1])
                self.index.add(matrix)
            
            self.save_index() # Auto-save
        except Exception as e:
            logger.error(f"Index build failed: {e}")
        
    def query(self, user_query: str):
        if self.doc_embeddings is None and self.index is None:
            return "Index not built.", ""
        
        logger.info(f"Processing RAG Query: {user_query}")
        
        try:
            q_res = genai.embed_content(
                model=self.embedding_model,
                content=user_query,
                task_type="retrieval_query"
            )
            q_embed = np.array([q_res['embedding']]).astype('float32')
            
            top_k = 5
            indices = []
            
            if self.index:
                D, I = self.index.search(q_embed, top_k)
                indices = I[0]
            else:
                scores = np.dot(self.doc_embeddings, q_embed.T).flatten()
                indices = scores.argsort()[-top_k:][::-1]
            
            context = ""
            for idx in indices:
                if idx < len(self.docs) and idx >= 0:
                    d = self.docs[idx]
                    context += f"- [r/{d['subreddit']}] {d['title']} (Score: {d['score']})\n"

            prompt = f"""
            Context from Subreddit Analysis:
            {context}
            
            User Query: {user_query}
            
            Answer based on the context provided. Cite specific posts if relevant.
            """
            resp = self.model.generate_content(prompt)
            return resp.text, context
            
        except Exception as e:
            logger.error(f"RAG Query failed: {e}")
            return f"Error: {e}", ""
