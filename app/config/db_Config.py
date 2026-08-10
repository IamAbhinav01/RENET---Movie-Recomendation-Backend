from sqlalchemy import create_engine
from server_Config import server_config

settings = server_config()
engine = create_engine(url=settings.DB_URL)

print(settings.PORT)