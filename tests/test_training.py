"""Verifica o fluxo completo com dados sintéticos e MLflow temporário."""

import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score

from water_potability.config import Settings
from water_potability.repository import FEATURES, TARGET
from water_potability.service import TrainingService


def test_training_registers_and_reloads_complete_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x, y = make_classification(n_samples=120, n_features=9, n_informative=5, random_state=7)
    data = pd.DataFrame(x, columns=FEATURES)
    data.loc[0, "ph"] = np.nan
    data[TARGET] = y
    data.to_csv("water.csv", index=False)
    settings = Settings(_env_file=None, data_path="water.csv", cv_folds=2, n_estimators=10)
    result = TrainingService(settings).execute()

    assert result["train_rows"] + result["test_rows"] == len(data)
    assert len(result["candidates"]) == 3
    selected = max(result["candidates"], key=lambda item: item["cv_f1_mean"])
    assert result["selected_model"] == selected["model"]
    report_dir = tmp_path / "reports" / result["run_id"]
    split = json.loads((report_dir / "split_indices.json").read_text())
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["train"]) | set(split["test"]) == set(range(len(data)))

    loaded = mlflow.sklearn.load_model(f"models:/{settings.model_name}/{result['model_version']}")
    predictions = pd.read_csv(report_dir / "test_predictions.csv")
    np.testing.assert_equal(loaded.predict(predictions[FEATURES]), predictions["predicted"])
    assert result["test_metrics"]["test_f1"] == f1_score(
        predictions["real"], predictions["predicted"]
    )
    assert (report_dir / "confusion_matrix.png").is_file()
    alias = MlflowClient().get_model_version_by_alias(settings.model_name, "selecionado")
    assert alias.version == result["model_version"]
    # Os demais candidatos só recebem métricas de validação.
    for candidate in result["candidates"]:
        if candidate["run_id"] != result["run_id"]:
            metrics = MlflowClient().get_run(candidate["run_id"]).data.metrics
            assert not any(name.startswith("test_") for name in metrics)
