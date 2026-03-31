from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGORA_")

    database_url: str = "sqlite+aiosqlite:///./agora.db"
    host: str = "127.0.0.1"
    port: int = 8321
    debug: bool = False
    cors_origins: list[str] = ["*"]


settings = Settings()
