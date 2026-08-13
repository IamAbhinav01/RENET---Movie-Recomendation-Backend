from app.config.artifacts_loader import load_models, models
from app.repository.interactionRepository import interaction_repo
from app.repository.itemsRepository import item_repo


def get_health_status():
    return {"status": "ok", "message": "ReNet Recommendation Engine is running"}


def reload_model_artifacts():
    models.clear()
    load_models()
    return {"status": "Models reloaded successfully"}


def get_interaction_history(user_id: int = 1):
    return interaction_repo.get_by_user(user_id)


def get_item_detail(item_id: int = 50):
    return item_repo.get_by_id(item_id)


# item_repo.update_poster(50, "https://poster.url/img.jpg", "A great movie...")
