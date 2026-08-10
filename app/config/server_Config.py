from dotenv import load_dotenv
from pydantic_settings import BaseSettings,SettingsConfigDict
from functools import lru_cache

load_dotenv()

class Settings(BaseSettings):
    APP_NAME :str = "RENET"
    DB_URL : str 
    PORT : int = 8000

    model_config = SettingsConfigDict(env_file='.env',env_file_encoding='utf-8',extra='allow')

@lru_cache
def server_config()->Settings:
    return Settings()