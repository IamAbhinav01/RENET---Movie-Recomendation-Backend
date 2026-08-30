import pickle as pkl
import faiss
import numpy as np
import lightgbm as lgbm
import traceback
import logging

models = {}
logger = logging.getLogger("artifacts_loader")


def cache_models():
    with open('app/artifacts/als_model.pkl','rb') as f:
        models["als"] = pkl.load(f)
    models["faiss_index"] = faiss.read_index("app/artifacts/content.index")
    models["content_item_ids"] = np.load("app/artifacts/content_item_ids.npy")

    # Load ranker (LightGBM). Be tolerant to corrupted/incompatible models so app can start.
    try:
        models["ranker"] = lgbm.Booster(model_file="app/artifacts/ranker.txt")
        logger.info("Loaded ranker model")
    except Exception as e:
        # keep app running even if ranker fails to load
        logger.warning(f"Failed to load ranker model: {e}")
        logger.debug(traceback.format_exc())
        models["ranker"] = None

def load_models():
    cache_models()
    return models

#output structure is like:
'''
{
    als:{
            model:''
            user_id_to_idx:'',
            idx_to_user_id:'',
            item_id_to_idx:'',
            idx_to_item_id:'',
            user_item_matrix:'',
    },
    
    faiss_index:'',
    content_item_ids:'',
    ranker:''
}


'''