import pickle as pkl
import faiss
import numpy as np
import lightgbm as lgbm


models= {}

def cache_models():
    with open('../artifacts/als_model.pkl','rb') as f:
        models["als"] = pkl.load(f)
    models["faiss_index"] = faiss.read_index("../artifacts/content.index")
    models["content_item_ids"] = np.load("../artifacts/content_item_ids.npy")    
    models["ranker"] = lgbm.Booster(model_file="../artifacts/ranker.txt")

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