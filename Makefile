POETRY ?= poetry

.PHONY: install train ui predict test lint format check package

install:
	$(POETRY) install

train:
	$(POETRY) run water-potability train

ui:
	$(POETRY) run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

predict:
	$(POETRY) run water-potability predict --input examples/amostras.csv --output reports/predicoes.csv

test:
	$(POETRY) run pytest -q

lint:
	$(POETRY) run ruff check .

format:
	$(POETRY) run ruff format .

check: lint test
