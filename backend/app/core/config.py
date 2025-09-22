from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Traffic Monitoring System"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str
    GEMINI_API_KEY: str | None = None  # Changed from OPENAI_API_KEY

    class Config:
        env_file = ".env"

settings = Settings()

