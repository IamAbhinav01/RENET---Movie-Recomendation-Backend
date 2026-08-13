from config.artifacts_loader import load_models
from config.db_Config import engine
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

embeddings = np.load("../artifacts/content_embeddings.npy").astype("float32")
items_id_to_row = {int(item_id): idx for idx, item_id in enumerate(content_item_ids)}

with open("../artifacts/ranker_features.json") as f:
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