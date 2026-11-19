from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./unplug.db"
    redis_url: str = "redis://localhost:6379/0"
    data_dir: Path = Path("./data")
    admin_emails: str = ""
    cf_access_team_domain: str = ""
    cf_access_aud: str = ""
    dev_user_email: str = ""
    max_queued_per_user: int = 5

    @property
    def admins(self) -> set[str]:
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}

    @property
    def songs_dir(self) -> Path:
        return self.data_dir / "songs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
