from typing import Generic, TypeVar, Type, List, Optional
from app.schemas.postgres_schema import Session
T = TypeVar("T")

class CrudRepository(Generic[T]):
    def __init__(self,model:Type[T]):
        self.model = model
    
    def get_all(self) -> List[T]:
            with Session() as session:
                return session.query(self.model).all()
    def get_by_id(self, id: int) -> Optional[T]:
        with Session() as session:
            return session.query(self.model).filter(self.model.id == id).first()
    def create(self, obj: T) -> T:
        with Session() as session:
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj
    def delete(self, id: int) -> Optional[T]:
        with Session() as session:
            obj = session.query(self.model).filter(self.model.id == id).first()
            if obj:
                session.delete(obj)
                session.commit()
            return obj