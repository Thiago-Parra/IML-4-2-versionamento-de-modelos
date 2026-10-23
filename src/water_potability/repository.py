"""Leitura e validação dos dados de entrada (padrão Repository)."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
TARGET = "Potability"


class CsvWaterRepository:
    """Centraliza o contrato do CSV usado no treino e na predição."""

    def __init__(self, path: Path):
        self.path = path

    def load(self, *, training: bool = True) -> pd.DataFrame:
        """Aceita atributos ausentes, mas exige rótulos válidos no treinamento."""
        data = pd.read_csv(self.path)
        required = FEATURES + ([TARGET] if training else [])
        missing = sorted(set(required) - set(data.columns))
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
        if data.empty:
            raise ValueError("O CSV não contém amostras.")

        # A seleção explícita também impede que o alvo seja usado como atributo.
        data = data.loc[:, required].copy()
        data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
        if np.isinf(data[FEATURES].to_numpy()).any():
            raise ValueError("Os atributos não podem conter valores infinitos.")
        if training:
            if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
                raise ValueError("Potability deve conter somente 0 ou 1, sem valores ausentes.")
            if data[TARGET].nunique() != 2:
                raise ValueError("O treinamento exige exemplos das duas classes.")
            if data[FEATURES].isna().all().any():
                raise ValueError("Há atributos sem nenhum valor observado.")
            data[TARGET] = data[TARGET].astype(int)
            # Duplicatas não podem aparecer em partições diferentes.
            data = data.drop_duplicates().reset_index(drop=True)
        return data
