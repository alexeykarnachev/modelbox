from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    triton_url: str = "127.0.0.1:8001"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
