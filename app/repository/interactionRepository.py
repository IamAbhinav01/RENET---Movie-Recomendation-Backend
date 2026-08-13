from typing import List
from app.repository.crudOperations import CrudRepository
from app.schemas.postgres_schema import Interactions, Session

class InteractionRepository(CrudRepository[Interactions]):
    def __init__(self):
        super().__init__(Interactions)

    def get_by_user(self, user_id: int) -> List[Interactions]:
        with Session() as session:
            return session.query(Interactions)\
                .filter(Interactions.user_id == user_id).all()

interaction_repo = InteractionRepository()
