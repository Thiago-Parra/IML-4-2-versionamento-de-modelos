"""Configurações validadas e carregadas do ambiente."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros compartilhados entre treinamento e predição."""

    model_config = SettingsConfigDict(
        env_prefix="WATER_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_path: Path = Path("water_potability.csv")
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "water-potability"
    model_name: str = "water-potability-classifier"
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)
    test_size: float = Field(default=0.2, gt=0, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    n_estimators: int = Field(default=100, ge=10, le=1000)
