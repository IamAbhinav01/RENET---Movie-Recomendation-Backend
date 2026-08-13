from app.repository.crudOperations import CrudRepository
from app.schemas.postgres_schema import Item, Session

class ItemRepository(CrudRepository[Item]):
    def __init__(self):
        super().__init__(Item)

    
    def update_poster_and_url(self, item_id: int, poster_url: str, plot: str):
        with Session() as session:
            item = session.query(Item).filter(Item.id == item_id).first()
            if item:
                item.poster_url = poster_url
                item.plot = plot
                session.commit()
            return item
    

item_repo = ItemRepository()
