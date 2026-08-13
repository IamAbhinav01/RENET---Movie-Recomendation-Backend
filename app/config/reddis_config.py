import redis
from app.config.server_Config import server_config
from app.config.logger_Config import setup_logger

settings = server_config()
logger = setup_logger()

try:
    client = redis.Redis(
        host=settings.HOST,
        port=settings.REDIS_PORT,
        decode_responses=True,
        protocol=2,
    )
    client.ping()
    logger.info("Connected to Redis")
except Exception as e:
    logger.warning(f"Redis unavailable, continuing without cache: {e}")
    client = None

