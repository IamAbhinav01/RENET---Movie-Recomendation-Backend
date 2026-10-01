from fastapi import APIRouter, Header, HTTPException

from app.services.operationService import (
    get_health_status,
    get_interaction_history,
    get_item_detail,
    reload_model_artifacts,
)
from app.services.recommendations import recommend, recommend_by_movie_name

router = APIRouter()


@router.get("/")
def health_status():
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
def recommend_movies(movie_name: str | None = None, n: int = 10):
    if not movie_name:
        raise HTTPException(status_code=400, detail="Provide a movie_name.")

    results = recommend_by_movie_name(movie_name, n=n)
    return {"movie_name": movie_name, "recommendations": results}


@router.get("/api/user/recommend")
def recommend_for_session_user(
    x_user_id: int = Header(alias="X-User-ID"), n: int = 10
):
    if x_user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid session user ID.")

    return {"user_id": x_user_id, "recommendations": recommend(x_user_id, n=n)}
