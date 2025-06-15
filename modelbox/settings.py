from pathlib import Path

from platformdirs import user_data_dir
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    triton_url: str = "127.0.0.1:8001"
    app_dir: Path = Path(user_data_dir("modelbox"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def media_dir(self) -> Path:
        return self.app_dir / "media"


settings = Settings()
