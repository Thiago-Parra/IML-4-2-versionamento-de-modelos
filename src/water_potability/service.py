"""Orquestração das etapas de treinamento, avaliação e versionamento."""

import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import __version__
from .config import Settings
from .models import ModelFactory
from .repository import FEATURES, TARGET, CsvWaterRepository
from .tracking import MlflowRepository

logger = logging.getLogger(__name__)


class TrainingService:
    """Coordena os componentes sem misturar entrada, modelo e persistência."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def execute(self) -> dict:
        """Seleciona pela validação cruzada e usa o teste apenas no escolhido."""
        settings = self.settings
        data = CsvWaterRepository(settings.data_path).load()
        x_train, x_test, y_train, y_test = train_test_split(
            data[FEATURES],
            data[TARGET],
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=data[TARGET],
        )
        if y_train.value_counts().min() < settings.cv_folds:
            raise ValueError("Há poucos exemplos por classe para a quantidade de folds.")

        tracking = MlflowRepository(settings)
        digest = hashlib.sha256(settings.data_path.read_bytes()).hexdigest()
        cv = StratifiedKFold(
            n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state
        )
        candidates = []
        for name in ("baseline", "logistic_regression", "random_forest"):
            model = ModelFactory.create(name, settings)
            with mlflow.start_run(run_name=f"{name}-{settings.n_estimators}") as run:
                mlflow.set_tags({"code_version": __version__, "dataset_sha256": digest})
                mlflow.log_params(
                    {
                        "model": name,
                        "random_state": settings.random_state,
                        "test_size": settings.test_size,
                        "cv_folds": settings.cv_folds,
                        "train_rows": len(x_train),
                        "test_rows": len(x_test),
                        "selection_metric": "cv_f1_mean",
                        "imputation": "median",
                        **{
                            k: v
                            for k, v in model.get_params().items()
                            if k.startswith("classifier__")
                        },
                    }
                )
                # Cada fold ajusta sua própria mediana e, quando usado, o escalonamento.
                scores = cross_validate(
                    model,
                    x_train,
                    y_train,
                    cv=cv,
                    n_jobs=1,
                    error_score="raise",
                    scoring={
                        "f1": "f1",
                        "roc_auc": "roc_auc",
                        "balanced_accuracy": "balanced_accuracy",
                    },
                )
                metrics = {}
                for metric in ("f1", "roc_auc", "balanced_accuracy"):
                    values = scores[f"test_{metric}"]
                    metrics[f"cv_{metric}_mean"] = float(values.mean())
                    metrics[f"cv_{metric}_std"] = float(values.std())
                mlflow.log_metrics(metrics)
                mlflow.log_dict({k: v.tolist() for k, v in scores.items()}, "validation/folds.json")
                candidates.append({"model": name, "run_id": run.info.run_id, **metrics})
                logger.info("%s | F1 médio na validação: %.4f", name, metrics["cv_f1_mean"])

        # O teste não participa da escolha do algoritmo nem dos hiperparâmetros.
        best = max(candidates, key=lambda candidate: candidate["cv_f1_mean"])
        model = ModelFactory.create(best["model"], settings)
        output = Path("reports") / best["run_id"]
        output.mkdir(parents=True, exist_ok=True)
        with mlflow.start_run(run_id=best["run_id"]):
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            probability = model.predict_proba(x_test)[:, 1]
            metrics = {
                "test_accuracy": accuracy_score(y_test, predicted),
                "test_balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                "test_precision": precision_score(y_test, predicted, zero_division=0),
                "test_recall": recall_score(y_test, predicted, zero_division=0),
                "test_f1": f1_score(y_test, predicted, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, probability),
            }
            mlflow.log_metrics(metrics)
            mlflow.set_tag("selected", "true")
            report = classification_report(
                y_test,
                predicted,
                labels=[0, 1],
                target_names=["nao_potavel", "potavel"],
                output_dict=True,
                zero_division=0,
            )
            matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
            display = ConfusionMatrixDisplay(matrix, display_labels=["Não potável", "Potável"])
            display.plot(cmap="Blues", colorbar=False)
            plt.title("Avaliação no conjunto de teste")
            plt.xlabel("Classe prevista")
            plt.ylabel("Classe real")
            plt.tight_layout()
            plt.savefig(output / "confusion_matrix.png", dpi=150)
            plt.close()

            # O exemplo contém todas as colunas e preserva os valores ausentes.
            model_uri, version = tracking.register(model, x_train.head(5))
            summary = {
                "code_version": __version__,
                "settings": settings.model_dump(mode="json"),
                "dataset_sha256": digest,
                "rows": len(data),
                "class_counts": data[TARGET].value_counts().sort_index().to_dict(),
                "missing_values": data.isna().sum().to_dict(),
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "candidates": candidates,
                "selected_model": best["model"],
                "run_id": best["run_id"],
                "model_uri": model_uri,
                "registered_model": settings.model_name,
                "model_version": version,
                "test_metrics": metrics,
                "confusion_matrix": matrix.tolist(),
            }
            self._write_json(output / "summary.json", summary)
            self._write_json(output / "classification_report.json", report)
            self._write_json(
                output / "split_indices.json",
                {"train": x_train.index.tolist(), "test": x_test.index.tolist()},
            )
            pd.DataFrame(candidates).to_csv(output / "validation.csv", index=False)
            predictions = x_test.copy()
            predictions["real"] = y_test
            predictions["predicted"] = predicted
            predictions["probability_potable"] = probability
            predictions.to_csv(output / "test_predictions.csv", index=True, index_label="row")
            mlflow.log_artifacts(str(output), artifact_path="evaluation")

        self._write_json(Path("reports/latest.json"), summary)
        logger.info(
            "Modelo %s registrado na versão %s. Relatórios: %s", best["model"], version, output
        )
        return summary

    @staticmethod
    def _write_json(path: Path, content: dict) -> None:
        """Grava relatórios legíveis para consulta sem abrir o MLflow."""
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
