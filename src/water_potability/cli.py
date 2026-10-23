"""Interface de linha de comando da aplicação."""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn

from .config import Settings
from .repository import FEATURES, CsvWaterRepository
from .service import TrainingService


def main() -> None:
    """Executa o treinamento ou utiliza uma versão registrada para prever."""
    parser = argparse.ArgumentParser(description="Classificação da potabilidade da água.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Treina, avalia e registra um modelo.")
    train.add_argument("--n-estimators", type=int, help="Quantidade de árvores da Random Forest.")
    predict = commands.add_parser("predict", help="Classifica amostras com um modelo registrado.")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("reports/predicoes.csv"))
    predict.add_argument("--model-uri", help="Exemplo: models:/water-potability-classifier/1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    overrides = {}
    if args.command == "train" and args.n_estimators is not None:
        overrides["n_estimators"] = args.n_estimators
    settings = Settings(**overrides)
    if args.command == "train":
        TrainingService(settings).execute()
    else:
        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_registry_uri(settings.tracking_uri)
        uri = args.model_uri or f"models:/{settings.model_name}@selecionado"
        model = mlflow.sklearn.load_model(uri)
        data = CsvWaterRepository(args.input).load(training=False)
        data["Potability"] = model.predict(data[FEATURES])
        data["probability_potable"] = model.predict_proba(data[FEATURES])[:, 1]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output, index=False)
        logging.info("Predições salvas em %s usando %s", args.output, uri)


if __name__ == "__main__":
    main()
