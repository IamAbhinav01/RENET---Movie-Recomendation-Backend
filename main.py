from app.schemas.postgres_schema import Base
from app.config.db_Config import engine
from app.config.artifacts_loader import load_models
import contextlib
from fastapi import FastAPI


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    load_models()
    yield


app = FastAPI(title="ReNet RecSys", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "ok", "message": "ReNet Recommendation Engine is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)