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

    # Fix CRLF line endings that corrupt the lightgbm model format on Windows
    model_path = "app/artifacts/ranker.txt"
    try:
        with open(model_path, "rb") as f:
            content = f.read()
        if b"\r\n" in content:
            content = content.replace(b"\r\n", b"\n")
            with open(model_path, "wb") as f:
                f.write(content)
            logger.info("Fixed CRLF line endings in ranker model")
    except Exception as e:
        logger.warning(f"Failed to check/fix line endings for ranker model: {e}")

    try:
        models["ranker"] = lgbm.Booster(model_file=model_path)
        logger.info("Loaded ranker model")
    except Exception as e:
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