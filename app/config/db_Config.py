from sqlalchemy import create_engine, text
from app.config.server_Config import server_config
from app.config.logger_Config import setup_logger


settings = server_config()
logger = setup_logger(name="RENET-DBCONFIG")

engine = create_engine(url=settings.DB_URL)

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("successfully connected to POSTGRESQL")
except Exception as e:
    logger.error(f"Error while connecting to the DB, error -> {e}")
