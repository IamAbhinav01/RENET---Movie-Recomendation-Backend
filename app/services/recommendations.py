from app.config.artifacts_loader import load_models
from app.config.db_Config import engine
from app.config.reddis_config import client
from sqlalchemy import text
import numpy as np
import json
import pandas as pd

models = load_models()
als_data = models['als']
als_model = als_data['model']
user_id_to_idx = als_data['user_id_to_idx']
idx_to_user_id = als_data['idx_to_user_id']
item_id_to_idx = als_data['item_id_to_idx']
idx_to_item_id = als_data['idx_to_item_id']
user_item_matrix = als_data['user_item_matrix']

faiss_index = models['faiss_index']
content_item_ids = models['content_item_ids']
ranker = models['ranker']


embeddings = np.load("app/artifacts/content_embeddings.npy").astype("float32")
item_id_to_row = {int(item_id): idx for idx, item_id in enumerate(content_item_ids)}

with open("app/artifacts/ranker_features.json") as f:
    feature_cols = json.load(f)

items_df = pd.read_sql("SELECT id, title, genres, primary_genre FROM items", engine)
item_genre = dict(zip(items_df["id"], items_df["primary_genre"]))
item_lookup = items_df.set_index("id")

interactions = pd.read_sql("SELECT item_id FROM interactions", engine)
popularity = interactions.groupby("item_id").size()
if len(popularity) > 0:
    popularity = (popularity / popularity.max()).to_dict()
else:
    popularity = {}



def get_user_interactions(user_id: int):
    """Retrieve user ratings from PostgreSQL database."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT item_id, rating FROM interactions WHERE user_id = :uid"),
            {"uid": user_id}
        )
        return pd.DataFrame(result.fetchall(), columns=["item_id", "rating"])
def get_als_candidates(user_id: int, n=100):
    user_idx = user_id_to_idx.get(user_id)
    if user_idx is None:
        return []
    
    ids, scores = als_model.recommend(
        user_idx,
        user_item_matrix[user_idx],
        N=n,
        filter_already_liked_items=True
    )
    return [{"item_id": idx_to_item_id[int(item_idx)], "als_score": float(score)} 
            for item_idx, score in zip(ids, scores)]
def get_content_candidates(positive_items, n=100):
    if not positive_items:
        return []
    
    rows = [item_id_to_row[item_id] for item_id in positive_items if item_id in item_id_to_row]
    if not rows:
        return []
    
    user_embedding = embeddings[rows].mean(axis=0)
    user_embedding = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)
    user_embedding = user_embedding.reshape(1, -1).astype("float32")
    
    scores, indices = faiss_index.search(user_embedding, n)
    
    candidates = []
    seen_items = set(positive_items)
    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue
        item_id = int(content_item_ids[index])
        if item_id in seen_items:
            continue
        candidates.append({"item_id": item_id, "content_sim": float(score)})
        
    return candidates
def merge_candidates(als_candidates, content_candidates):
    merged = {}
    for candidate in als_candidates:
        item_id = candidate["item_id"]
        merged[item_id] = {"item_id": item_id, "als_score": candidate["als_score"], "content_sim": 0.0}
    for candidate in content_candidates:
        item_id = candidate["item_id"]
        if item_id not in merged:
            merged[item_id] = {"item_id": item_id, "als_score": 0.0, "content_sim": candidate["content_sim"]}
        else:
            merged[item_id]["content_sim"] = candidate["content_sim"]
    return list(merged.values())
def rank_candidates(candidates, user_genres):
    if not candidates:
        return []
    
    rows = []
    for candidate in candidates:
        item_id = candidate["item_id"]
        genre = item_genre.get(item_id, "Unknown")
        genre_match = 1.0 if genre in user_genres else 0.0
        rows.append({
            "item_id": item_id,
            "als_score": candidate.get("als_score", 0.0),
            "content_sim": candidate.get("content_sim", 0.0),
            "popularity": popularity.get(item_id, 0.0),
            "genre_match": genre_match
        })
        
    features = pd.DataFrame(rows)
    predictions = ranker.predict(features[feature_cols])
    features["ranker_score"] = predictions
    features = features.sort_values("ranker_score", ascending=False).reset_index(drop=True)
    return features.to_dict(orient="records")
def rerank(ranked_candidates, all_watched_items, max_per_genre=3, top_k=10):
    seen_genres = {}
    final = []
    for candidate in ranked_candidates:
        item_id = candidate["item_id"]
        if item_id in all_watched_items:
            continue
        genre = item_genre.get(item_id, "Unknown")
        if seen_genres.get(genre, 0) >= max_per_genre:
            continue
        seen_genres[genre] = seen_genres.get(genre, 0) + 1
        final.append(candidate)
        if len(final) >= top_k:
            break
    return final
def recommend(user_id: int, n=10):
    # 1. Check Redis Cache First
    cache_key = f"recs:user:{user_id}"
    cached_data = client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    # 2. Pipeline Run
    user_interactions = get_user_interactions(user_id)
    all_watched_items = set(user_interactions["item_id"])
    positive_items = set(user_interactions[user_interactions["rating"] >= 4.0]["item_id"])
    user_genres = {item_genre.get(item_id, "Unknown") for item_id in positive_items}
    
    als_candidates = get_als_candidates(user_id, n=100)
    content_candidates = get_content_candidates(list(positive_items), n=100)
    candidates = merge_candidates(als_candidates, content_candidates)
    
    ranked_candidates = rank_candidates(candidates, user_genres)
    if not ranked_candidates:
        return []
        
    final_candidates = rerank(ranked_candidates, all_watched_items, top_k=n)
    
    results = []
    for candidate in final_candidates:
        item_id = candidate["item_id"]
        if item_id not in item_lookup.index:
            continue
        movie = item_lookup.loc[item_id]
        results.append({
            "id": int(item_id),
            "title": movie["title"],
            "genres": movie["genres"],
            "score": float(candidate["ranker_score"])
        })
        
    # 3. Cache the output in Redis for 1 hour
    client.setex(cache_key, 3600, json.dumps(results))
    return results