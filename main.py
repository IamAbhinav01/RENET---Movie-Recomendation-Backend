import contextlib
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.artifacts_loader import load_models
from app.config.db_Config import engine
from app.config.server_Config import server_config
from app.router.operations import router
from app.schemas.postgres_schema import Base


settings = server_config()
PORT = settings.PORT


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    load_models()
    yield


app = FastAPI(title="ReNet Recommendation System", version="1.0.0", lifespan=lifespan)
frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=True)