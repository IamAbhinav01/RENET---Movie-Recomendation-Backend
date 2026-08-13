import redis
from server_Config import server_config
from logger_Config import setup_logger

settings = server_config()
logger = setup_logger()

try:
    client = redis.Redis(host=settings.HOST,port=settings.REDIS_PORT,decode_responses=True)
    logger.info("Connected to Redis")

except Exception as e:
    logger.error(f"Error while conneting to redis , error -> {e}")

