"""Persistência dos experimentos e modelos no MLflow."""

from importlib.metadata import version as package_version

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from .config import Settings


class MlflowRepository:
    """Isola a configuração e o registro do modelo no serviço externo."""

    def __init__(self, settings: Settings):
        self.settings = settings
        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_registry_uri(settings.tracking_uri)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(settings.experiment_name)
        if experiment is None:
            # Caminho relativo permite reabrir a entrega em outra pasta, a partir da raiz.
            client.create_experiment(
                settings.experiment_name, artifact_location="./mlartifacts/water-potability"
            )
        mlflow.set_experiment(settings.experiment_name)

    def register(self, model: Pipeline, example: pd.DataFrame) -> tuple[str, str]:
        """Salva o pipeline completo e cria uma nova versão no Model Registry."""
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            # Formato explícito para manter a serialização consistente entre versões do MLflow.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=[
                f"{package}=={package_version(package)}"
                for package in ("mlflow", "scikit-learn", "pandas", "numpy", "cloudpickle")
            ],
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
        )
        version = mlflow.register_model(info.model_uri, self.settings.model_name)
        # O alias representa o último modelo escolhido pela validação desta execução.
        MlflowClient().set_registered_model_alias(
            self.settings.model_name, "selecionado", version.version
        )
        return info.model_uri, version.version
