from config.artifacts_loader import load_models

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