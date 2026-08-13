import contextlib

from fastapi import FastAPI

from app.config.artifacts_loader import load_models
from app.config.db_Config import engine
from app.router.operations import router
from app.schemas.postgres_schema import Base


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    load_models()
    yield


app = FastAPI(title="ReNet Recommendation System", version="1.0.0", lifespan=lifespan)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)