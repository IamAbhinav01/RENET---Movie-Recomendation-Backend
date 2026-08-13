from fastapi import APIRouter

from app.services.operationService import (
    get_health_status,
    get_interaction_history,
    get_item_detail,
    reload_model_artifacts,
)
from app.services.recommendations import recommend

router = APIRouter()


@router.get("/")
def root():
    return get_health_status()


@router.post("/admin/reload-models")
def reload_models():
    return reload_model_artifacts()


@router.get("/interaction")
def interaction_endpoint():
    return get_interaction_history()


@router.get("/item")
def item_endpoint():
    return get_item_detail()


@router.get("/api/recommend")
def recommend_movies(user_id: int, n: int = 10):
    return {"user_id": user_id, "recommendations": recommend(user_id, n=n)}
