"""Testes dos contratos de dados e do pré-processamento."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.base import clone

from water_potability.config import Settings
from water_potability.models import ModelFactory
from water_potability.repository import FEATURES, TARGET, CsvWaterRepository


def sample_data():
    """Monta exemplos pequenos sem depender do dataset externo."""
    data = pd.DataFrame({feature: [1.0, 2.0, 3.0, 4.0] for feature in FEATURES})
    data[TARGET] = [0, 1, 0, 1]
    data.loc[0, "ph"] = np.nan
    return data


def test_repository_accepts_missing_features_and_removes_duplicates(tmp_path):
    data = sample_data()
    path = tmp_path / "water.csv"
    pd.concat([data, data.iloc[[1]]]).to_csv(path, index=False)
    loaded = CsvWaterRepository(path).load()
    assert len(loaded) == 4
    assert loaded["ph"].isna().sum() == 1


@pytest.mark.parametrize("invalid", [np.nan, 2, -1, "potavel"])
def test_repository_rejects_invalid_target(tmp_path, invalid):
    data = sample_data()
    data[TARGET] = data[TARGET].astype(object)
    data.loc[0, TARGET] = invalid
    path = tmp_path / "water.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Potability"):
        CsvWaterRepository(path).load()


def test_prediction_does_not_require_or_use_target(tmp_path):
    path = tmp_path / "samples.csv"
    sample_data().to_csv(path, index=False)
    assert list(CsvWaterRepository(path).load(training=False)) == FEATURES


def test_repository_rejects_missing_column(tmp_path):
    path = tmp_path / "water.csv"
    sample_data().drop(columns="Sulfate").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Colunas obrigatórias"):
        CsvWaterRepository(path).load()


@pytest.mark.parametrize("name", ["baseline", "logistic_regression", "random_forest"])
def test_pipeline_imputes_from_training_only(name):
    model = ModelFactory.create(name, Settings(_env_file=None, n_estimators=10))
    train = sample_data()
    # Um valor extremo na predição não deve alterar a mediana aprendida no treino.
    model.fit(train[FEATURES], train[TARGET])
    before = model.named_steps["imputer"].statistics_.copy()
    prediction = train[FEATURES].copy()
    prediction.loc[1, "ph"] = 10000
    assert len(model.predict(prediction)) == 4
    np.testing.assert_equal(model.named_steps["imputer"].statistics_, before)
    assert before[0] == 3.0
    assert not hasattr(clone(model).named_steps["imputer"], "statistics_")


def test_settings_validate_parameters():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, test_size=0.8)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cv_folds=1)
