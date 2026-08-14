from fastapi import APIRouter, HTTPException

from app.services.operationService import (
    get_health_status,
    get_interaction_history,
    get_item_detail,
    reload_model_artifacts,
)
from app.services.recommendations import recommend, recommend_by_movie_name

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
def recommend_movies(movie_name: str | None = None, user_id: int | None = None, n: int = 10):
    if movie_name:
        results = recommend_by_movie_name(movie_name, n=n)
        return {"movie_name": movie_name, "recommendations": results}

    if user_id is not None:
        return {"user_id": user_id, "recommendations": recommend(user_id, n=n)}

    raise HTTPException(status_code=400, detail="Provide either movie_name or user_id.")
