from app.repository.crudOperations import CrudRepository
from app.schemas.postgres_schema import User

class UserRepository(CrudRepository[User]):
    def __init__(self):
        super().__init__(User)


user_repo = UserRepository()
