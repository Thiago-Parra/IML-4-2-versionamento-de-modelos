"""Criação dos pipelines de classificação (padrão Factory)."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings


class ModelFactory:
    """Constrói modelos com a mesma interface de treinamento e predição."""

    @staticmethod
    def create(name: str, settings: Settings) -> Pipeline:
        """Mantém todo o pré-processamento dentro do pipeline de cada fold."""
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if name == "baseline":
            # Prevê as classes conforme a distribuição do treino; serve como referência.
            classifier = DummyClassifier(strategy="stratified", random_state=settings.random_state)
        elif name == "logistic_regression":
            steps.append(("scaler", StandardScaler()))
            classifier = LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=settings.random_state
            )
        elif name == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=settings.n_estimators,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=settings.random_state,
                n_jobs=1,
            )
        else:
            raise ValueError(f"Modelo desconhecido: {name}")
        return Pipeline([*steps, ("classifier", classifier)])
