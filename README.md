## Autor: Thiago Henrique Parra

# IML4.2 — Versionamento de Modelos de Aprendizado de Máquina

Classificação da potabilidade da água com scikit-learn, tracking de experimentos e registro de versões no MLflow. Atividade da disciplina Ciclo de Vida do Modelo, MBA em Machine Learning in Production — UFSCar.

## Dados e objetivo

O arquivo `water_potability.csv`, mantido na raiz, contém 3.276 amostras, nove atributos numéricos e o alvo `Potability`: `0` indica água não potável e `1` indica água potável. São 1.998 exemplos da classe 0 e 1.278 da classe 1. Há 491 valores ausentes em `ph`, 781 em `Sulfate` e 162 em `Trihalomethanes`. Não foram encontradas linhas duplicadas no arquivo fornecido.

Fonte: [Water Quality — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability/data). 

| Coluna | Medida |
|---|---|
| `ph` | pH |
| `Hardness` | Dureza |
| `Solids` | Sólidos dissolvidos |
| `Chloramines` | Cloraminas |
| `Sulfate` | Sulfato |
| `Conductivity` | Condutividade |
| `Organic_carbon` | Carbono orgânico |
| `Trihalomethanes` | Trihalometanos |
| `Turbidity` | Turbidez |

## Organização e padrões de projeto


```text
src/water_potability/
  cli.py          # Comandos train e predict
  config.py       # Configurações com Pydantic Settings
  repository.py   # Repository: leitura e validação do CSV
  models.py       # Factory: criação dos pipelines
  tracking.py     # Repository: integração com o MLflow
  service.py      # Service Layer: coordenação do treinamento
tests/            # Testes automatizados
examples/         # Amostras de entrada para predição
reports/          # Evidências de validação, teste e versões
water_potability.csv
pyproject.toml
poetry.lock
Makefile
```

**Factory:** `ModelFactory.create` centraliza a construção dos três pipelines. **Repository:** `CsvWaterRepository` trata a entrada de dados e `MlflowRepository` concentra a configuração e o registro no MLflow. **Service Layer:** `TrainingService` coordena as etapas.

## Instalação

Requisitos: Python 3.11 a 3.13 e Poetry 2.4.3. O Make é opcional. Execute os comandos a partir da raiz do projeto; os caminhos dos dados e do MLflow são relativos a essa pasta.

```powershell
poetry install
poetry run water-potability --help
```

As configurações têm valores padrão e podem ser alteradas com variáveis `WATER_` ou copiando `.env.example` para `.env`.

| Configuração | Padrão |
|---|---|
| `WATER_DATA_PATH` | `water_potability.csv` |
| `WATER_TRACKING_URI` | `sqlite:///mlflow.db` |
| `WATER_EXPERIMENT_NAME` | `water-potability` |
| `WATER_MODEL_NAME` | `water-potability-classifier` |
| `WATER_RANDOM_STATE` | `42` |
| `WATER_TEST_SIZE` | `0.2` |
| `WATER_CV_FOLDS` | `5` |
| `WATER_N_ESTIMATORS` | `100` |

## Treinamento e validação

```powershell
poetry run water-potability train
```

O processo realiza as seguintes etapas:

1. Valida colunas, tipos numéricos e rótulos; remove eventuais linhas duplicadas antes da divisão.
2. Separa 80% para treinamento e 20% para teste, com estratificação e semente 42.
3. Compara um `DummyClassifier` estratificado, regressão logística e Random Forest por validação cruzada estratificada com cinco folds, somente no treino.
4. Seleciona o maior F1 médio da classe potável. Em empate, mantém o primeiro candidato na ordem apresentada.
5. Ajusta o pipeline escolhido em todo o treino e avalia uma vez no teste reservado.
6. Registra o pipeline completo, sua assinatura, exemplo de entrada e uma nova versão no Model Registry.

A referência estratificada prevê conforme as proporções do treino. A regressão logística usa padronização e pesos de classes balanceados. A Random Forest usa 100 árvores, mínimo de duas amostras por folha e pesos balanceados. Os hiperparâmetros são fixos; não há busca automática.

A imputação pela mediana fica dentro de cada `Pipeline`, assim como o `StandardScaler` da regressão logística. Cada fold aprende essas transformações somente na sua partição de treinamento. O conjunto de teste não participa da escolha do algoritmo. A coluna `Potability` nunca entra nos atributos.

O F1 combina precisão e recall da classe 1, relevante para o desbalanceamento observado. Também são registrados ROC AUC e acurácia balanceada na validação; no teste, acurácia, acurácia balanceada, precisão, recall e F1, além de ROC AUC, relatório por classe e matriz de confusão. As previsões usam o comportamento padrão dos classificadores, sem ajuste do limiar pelo teste.

## Tracking e registro no MLflow

O treinamento funciona sem iniciar um servidor. Os metadados ficam em `mlflow.db` (SQLite) e os arquivos em `mlartifacts/`. Cada execução cria três runs, um por candidato, e registra uma versão do selecionado.

Para abrir a interface em outro terminal, na raiz do projeto:

```powershell
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Acesse <http://127.0.0.1:5000>. Em **Experiments**, selecione `water-potability` e compare `cv_f1_mean`, parâmetros e artefatos. O run selecionado tem a tag `selected=true` e as métricas `test_*`. Em **Models**, abra `water-potability-classifier` para ver as versões e seus runs de origem.

Para gerar outra versão com uma configuração diferente:

```powershell
poetry run water-potability train --n-estimators 200
```

O registro incrementa a versão do modelo e move o alias `selecionado` para a versão escolhida nessa execução. Esse alias indica a última seleção pela validação; não compara com o histórico nem garante melhoria. As métricas de teste não devem ser usadas para continuar ajustando os modelos.

Além dos parâmetros e métricas, cada run contém a versão do código e o SHA-256 do CSV. O run escolhido também contém os índices das partições, previsões do teste, relatório por classe, tabela da validação e matriz de confusão. `reports/latest.json` aponta os resultados da última execução e as subpastas de `reports/` preservam as evidências anteriores.

As duas versões já executadas e suas métricas estão documentadas em [reports/README.md](reports/README.md).

## Predição com uma versão registrada

```powershell
poetry run water-potability predict --input examples/amostras.csv --output reports/predicoes.csv
```

Por padrão, carrega o alias `selecionado`. Para reproduzir uma versão específica:

```powershell
poetry run water-potability predict --input examples/amostras.csv --model-uri models:/water-potability-classifier/1 --output reports/predicoes_v1.csv
```

O CSV de entrada deve ter as nove colunas de atributos. Valores ausentes são tratados pelo pipeline salvo. A saída contém os atributos, `Potability` prevista e `probability_potable`. Os exemplos são amostras do dataset, para demonstrar o comando.

## Qualidade e versionamento do código

```powershell
poetry check --lock
poetry run ruff check .
poetry run ruff format --check .
poetry run pytest -q
```

Os testes verificam a validação do CSV, rejeição de rótulos inválidos, remoção de duplicatas, predição sem o alvo e preservação das estatísticas do pré-processamento após a predição. O workflow do GitHub Actions executa as mesmas verificações em Python 3.11.

Instale os hooks locais:

```powershell
poetry run pre-commit install
poetry run pre-commit run --all-files
```

O Makefile oferece `install`, `train`, `ui`, `predict`, `test`, `lint`, `format` e `check`.

O Bump2version atualiza a versão da aplicação em `pyproject.toml` e `__init__.py`, sem criar commits ou tags:

```powershell
poetry run bump2version patch
poetry lock
```

A versão da aplicação e a versão do modelo são independentes: a primeira identifica o código; o MLflow gera uma versão numérica a cada registro de modelo. Mudanças de versão da aplicação devem ser feitas quando houver uma nova revisão do código.