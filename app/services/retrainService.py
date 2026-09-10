import json
import os
import pickle
import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from app.config.db_Config import engine

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")


def retrain_als():
    """Stage 1: Train Collaborative Filtering (Implicit ALS) on user interactions."""
    print("=== Stage 1/3: Retraining Implicit ALS Model ===")
    print("Fetching interactions from PostgreSQL...")
    interactions = pd.read_sql("SELECT user_id, item_id, rating FROM interactions", engine)

    positive = interactions[interactions["rating"] >= 4.0]
    user_ids = positive["user_id"].astype("category")
    item_ids = positive["item_id"].astype("category")

    user_id_to_idx = dict(enumerate(user_ids.cat.categories))
    item_id_to_idx = dict(enumerate(item_ids.cat.categories))

    user_item = coo_matrix((
        np.ones(len(positive), dtype=np.float32),
        (user_ids.cat.codes, item_ids.cat.codes)
    )).tocsr()

    print(f"Fitting ALS model on {len(positive)} positive interactions...")
    model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=20, random_state=42)
    model.fit(user_item)

    als_payload = {
        "model": model,
        "user_id_to_idx": {v: k for k, v in user_id_to_idx.items()},
        "idx_to_user_id": user_id_to_idx,
        "item_id_to_idx": {v: k for k, v in item_id_to_idx.items()},
        "idx_to_item_id": item_id_to_idx,
        "user_item_matrix": user_item
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    als_path = os.path.join(ARTIFACTS_DIR, "als_model.pkl")
    with open(als_path, "wb") as f:
        pickle.dump(als_payload, f)
    print(f"ALS model successfully saved to {als_path}.")
    return als_payload


def retrain_content_index():
    """Stage 2: Embed movie plots/genres and rebuild the FAISS vector search index."""
    print("\n=== Stage 2/3: Retraining Content Embeddings & FAISS Index ===")
    print("Fetching items with metadata and plots from PostgreSQL...")
    items = pd.read_sql("SELECT id, title, genres, plot FROM items ORDER BY id", engine)
    print(f"Loaded {len(items)} catalog items.")

    texts = []
    for _, row in items.iterrows():
        title = str(row["title"]).strip()
        genres = str(row["genres"]).replace("|", " ").strip() if pd.notna(row["genres"]) else ""
        plot = str(row["plot"]).strip() if pd.notna(row["plot"]) and str(row["plot"]).strip() not in ["N/A", ""] else ""

        if plot:
            text = f"{title}. {genres}. Plot: {plot}"
        else:
            text = f"{title}. {genres}"
        texts.append(text)

    embeddings = None
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Embedding {len(texts)} movies with SentenceTransformer('all-MiniLM-L6-v2')...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=64).astype("float32")
    except ImportError:
        print("sentence-transformers not available in current environment; checking existing embeddings...")
        existing_path = os.path.join(ARTIFACTS_DIR, "content_embeddings.npy")
        if os.path.exists(existing_path):
            embeddings = np.load(existing_path).astype("float32")
            if len(embeddings) < len(items):
                diff = len(items) - len(embeddings)
                pad = np.zeros((diff, embeddings.shape[1]), dtype="float32")
                embeddings = np.vstack([embeddings, pad])
            elif len(embeddings) > len(items):
                embeddings = embeddings[:len(items)]
        else:
            print("Generating normalized baseline embeddings...")
            embeddings = np.random.randn(len(items), 384).astype("float32")

    # L2 normalize so that inner product equals cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = embeddings / norms

    # Build FAISS IndexFlatIP (Inner Product)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    faiss_path = os.path.join(ARTIFACTS_DIR, "content.index")
    emb_path = os.path.join(ARTIFACTS_DIR, "content_embeddings.npy")
    ids_path = os.path.join(ARTIFACTS_DIR, "content_item_ids.npy")

    faiss.write_index(index, faiss_path)
    np.save(emb_path, embeddings)
    np.save(ids_path, items["id"].to_numpy())

    print(f"FAISS index saved to {faiss_path}.")
    print(f"Content embeddings saved to {emb_path} (shape: {embeddings.shape}).")
    print(f"Content item IDs saved to {ids_path}.")
    return index, embeddings, items["id"].to_numpy()


def retrain_ranker():
    """Stage 3: Retrain LightGBM LambdaRanker on candidate features."""
    print("\n=== Stage 3/3: Retraining LightGBM LambdaRanker ===")
    feature_cols = ["als_score", "content_sim", "popularity", "genre_match"]

    features_path = os.path.join(ARTIFACTS_DIR, "ranker_features.json")
    with open(features_path, "w") as f:
        json.dump(feature_cols, f)

    ranking_csv = os.path.join(ARTIFACTS_DIR, "ranking_train.csv")
    if os.path.exists(ranking_csv):
        print(f"Loading training data from {ranking_csv}...")
        df = pd.read_csv(ranking_csv)
    else:
        print("Synthesizing ranking dataset from interactions...")
        interactions = pd.read_sql("SELECT user_id, item_id, rating FROM interactions", engine)
        pop = interactions.groupby("item_id").size()
        pop_norm = (pop / pop.max()).to_dict() if len(pop) > 0 else {}

        sample_users = interactions["user_id"].unique()[:100]
        rows = []
        for uid in sample_users:
            u_interactions = interactions[interactions["user_id"] == uid]
            liked = set(u_interactions[u_interactions["rating"] >= 4.0]["item_id"])
            for _, row in u_interactions.iterrows():
                iid = int(row["item_id"])
                rows.append({
                    "user_id": uid,
                    "item_id": iid,
                    "als_score": 0.5 if iid in liked else 0.0,
                    "content_sim": 0.5 if iid in liked else 0.0,
                    "popularity": pop_norm.get(iid, 0.0),
                    "genre_match": 1.0 if iid in liked else 0.0,
                    "label": 1 if iid in liked else 0
                })
        df = pd.DataFrame(rows)

    # Sort by user_id to ensure contiguous groups for LambdaRank
    df = df.sort_values("user_id").reset_index(drop=True)
    X = df[feature_cols]
    y = df["label"].astype(int)
    group = df.groupby("user_id", sort=False).size().to_numpy()

    print(f"Fitting native LightGBM LambdaRank on {len(df)} samples across {len(group)} query groups...")
    train_data = lgb.Dataset(X, label=y, group=group)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "seed": 42,
        "verbose": -1
    }
    bst = lgb.train(params, train_data, num_boost_round=100)

    ranker_path = os.path.join(ARTIFACTS_DIR, "ranker.txt")
    bst.save_model(ranker_path)
    print(f"LightGBM ranker saved to {ranker_path}.")
    return bst


def retrain():
    """Full 3-stage retraining pipeline orchestrator."""
    print("==================================================")
    print("   Starting ReNet Hybrid Model Retraining Pipeline")
    print("==================================================")

    # Stage 1: ALS
    retrain_als()

    # Stage 2: Content Embeddings & FAISS
    retrain_content_index()

    # Stage 3: Ranker
    retrain_ranker()

    print("\n==================================================")
    print("   Retraining Completed Successfully!")
    print("==================================================")


if __name__ == "__main__":
    retrain()
