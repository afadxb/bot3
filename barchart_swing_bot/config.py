from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    mysql_url: str = Field("sqlite:///./app.db", env="MYSQL_URL")
    tz: str = Field("America/Toronto", env="TZ")
    pushover_user: str | None = Field(default=None, env="PUSHOVER_USER")
    pushover_token: str | None = Field(default=None, env="PUSHOVER_TOKEN")
    scrape_allowed: bool = Field(False, env="SCRAPE_ALLOWED")
    risk_envelope: str = Field("on", env="RISK_ENVELOPE")
    api_token: str = Field("changeme", env="API_TOKEN")
    premarket_min_available: float = Field(1000.0, env="PREMARKET_MIN_AVAILABLE")
    paper_start_balance: float = Field(100000.0, env="PAPER_START_BALANCE")
    max_daily_loss_pct: float = Field(2.0, env="MAX_DAILY_LOSS_PCT")
    volume_multiple: float = Field(1.5, env="VOLUME_MULTIPLE")
    atr_lambda: float = Field(1.0, env="ATR_LAMBDA")
    epsilon_ticks: float = Field(0.01, env="EPSILON_TICKS")

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings()
