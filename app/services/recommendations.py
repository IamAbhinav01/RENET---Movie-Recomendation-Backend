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


def _load_items_from_db():
    try:
        return pd.read_sql("SELECT id, title, genres, primary_genre FROM items", engine)
    except Exception:
        return None


def load_items():
    """Ensure `items_df`, `item_genre`, `item_lookup`, and `popularity` are populated.
    Falls back to the bundled CSV if the DB table is missing."""
    global items_df, item_genre, item_lookup, popularity
    if items_df is not None and not items_df.empty:
        return

    df = _load_items_from_db()
    if df is None:
        try:
            csv = pd.read_csv("app/dataset/ml-latest-small/movies.csv")
            csv = csv.rename(columns={"movieId": "id", "genres": "genres", "title": "title"})
            csv["primary_genre"] = csv["genres"].apply(lambda g: g.split("|")[0] if pd.notna(g) and g != "" else "Unknown")
            items_df = csv[["id", "title", "genres", "primary_genre"]]
        except Exception:
            items_df = pd.DataFrame(columns=["id", "title", "genres", "primary_genre"])
    else:
        items_df = df

    item_genre = dict(zip(items_df["id"], items_df["primary_genre"]))
    item_lookup = items_df.set_index("id")

    try:
        interactions = pd.read_sql("SELECT item_id FROM interactions", engine)
        pop = interactions.groupby("item_id").size()
        if len(pop) > 0:
            popularity = (pop / pop.max()).to_dict()
        else:
            popularity = {}
    except Exception:
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
def get_movie_id_by_name(movie_name: str):
    load_items()
    if not movie_name or not isinstance(movie_name, str):
        return None

    normalized = movie_name.strip().lower()
    if not normalized:
        return None

    matches = item_lookup[item_lookup["title"].str.lower().str.contains(normalized, case=False, na=False)]
    if matches.empty:
        exact = item_lookup[item_lookup["title"].str.lower() == normalized]
        if exact.empty:
            return None
        return int(exact.index[0])

    return int(matches.index[0])


def recommend_by_movie_name(movie_name: str, n=10):
    load_items()
    movie_id = get_movie_id_by_name(movie_name)
    if movie_id is None:
        return []

    positive_items = [movie_id]
    content_candidates = get_content_candidates(positive_items, n=100)
    ranked_candidates = rank_candidates(content_candidates, {item_genre.get(movie_id, "Unknown")})
    if not ranked_candidates:
        return []

    final_candidates = rerank(ranked_candidates, {movie_id}, top_k=n)
    results = []
    for candidate in final_candidates:
        item_id = candidate["item_id"]
        if item_id == movie_id or item_id not in item_lookup.index:
            continue
        movie = item_lookup.loc[item_id]
        results.append({
            "id": int(item_id),
            "title": movie["title"],
            "genres": movie["genres"],
            "score": float(candidate["ranker_score"])
        })
    return results


def recommend(user_id: int, n=10):
    load_items()
    # 1. Check Redis Cache First
    cache_key = f"recs:user:{user_id}:n:{n}"
    if client is not None:
        try:
            cached_data = client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception:
            pass

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

    # 3. Cache the output in Redis for 1 hour if available
    if client is not None:
        try:
            client.setex(cache_key, 3600, json.dumps(results))
        except Exception:
            pass
    return results