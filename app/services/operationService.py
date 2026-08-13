from app.repository.itemsRepository import item_repo
from app.repository.interactionRepository import interaction_repo


history = interaction_repo.get_by_user(1)


movie = item_repo.get_by_id(50)

print(history)
print(movie)
# item_repo.update_poster(50, "https://poster.url/img.jpg", "A great movie...")
