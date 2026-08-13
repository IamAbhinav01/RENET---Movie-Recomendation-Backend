from sqlalchemy import create_engine, text
from server_Config import server_config
from logger_Config import setup_logger
from schemas.postgres_schema import Base

settings = server_config()
logger = setup_logger(name="RENET-DBCONFIG")

engine = create_engine(url=settings.DB_URL)

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("successfully connected to POSTGRESQL")
    Base.metadata.create_all(bind = engine)
    logger.info("Database tables ready.")
except Exception as e:
    logger.error(f"Error while connecting to the DB, error -> {e}")
