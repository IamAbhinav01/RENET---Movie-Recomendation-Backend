import pickle
import faiss
import numpy as np
import pandas as pd
import lightgbm as lgb
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from app.config.db_Config import engine

ARTIFACTS_DIR = "app/artifacts"

def retrain():
    print("Fetching interactions from DB...")
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
    
    print("Retraining ALS...")
    model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=20)
    model.fit(user_item)
    
    als_payload = {
        "model": model,
        "user_id_to_idx": {v: k for k, v in user_id_to_idx.items()},
        "idx_to_user_id": user_id_to_idx,
        "item_id_to_idx": {v: k for k, v in item_id_to_idx.items()},
        "idx_to_item_id": item_id_to_idx,
        "user_item_matrix": user_item
    }
    
    with open(f"{ARTIFACTS_DIR}/als_model.pkl", "wb") as f:
        pickle.dump(als_payload, f)
    print("ALS model saved.")

if __name__ == "__main__":
    retrain()
