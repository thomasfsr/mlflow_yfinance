# MLflow + YFinance – Monitoramento de Pipeline de Previsão de Preços

Este repositório reúne um workflow para treinar e monitorar modelos de previsão de preços de ações utilizando MLflow em conjunto com dados históricos obtidos via YFinance. O foco é demonstrar como acompanhar métricas, experimentos e re-treinar modelos com uma configuração de time series.

## Visão Geral
- Objetivo: fornecer uma aplicação prática para monitorar o pipeline de um modelo que prevê o preço de fechamento (Close) de ativos disponíveis na biblioteca YFinance.
- Dados: séries temporais de ações com foco na coluna Close e Volume; utiliza lag features para criar janelas temporais para treinamento.
- Modelos: XGBoost e LightGBM (regressores de boosting em árvore), utilizados em um esquema de MultiOutputRegressor para prever múltiplas saídas simultaneamente.
- Monitoramento: MLflow é utilizado para registrar hiperparâmetros, métricas (RMSE) e artefatos do experimento, facilitando comparação entre runs.

## Tecnologias Principais
- Python 3.11
- MLflow
- YFinance (dados financeiros)
- XGBoost
- LightGBM
- Scikit-learn
- Pandas, NumPy
- Taskipy (para tarefas de linha de comando via Poetry)

## Estrutura do Projeto
- src/: código-fonte do projeto
  - finance_get.py: utilitários para download, suavização, criação de lag e empacotamento de datasets para treino/validação.
  - model_ensemble.py: implementação do pipeline de treino/modelo com XGBoost/LightGBM, validação em séries temporais e integração com MLflow.
  - script_mlflow.py: script principal (invocado via Taskipy) para orquestrar experiments.
  - try.py: helpers/adicionais (se aplicável).
- pyproject.toml: configuração do Poetry com dependências e tarefas (run, mlui).
- README.md: este documento.

## Instalação
Pré-requisitos: Python 3.11+ e Poetry.

1. Clone o repositório.
2. Instale as dependências com Poetry:
   - poetry install
3. Ative o ambiente e instale dependências: poetry shell

Nota: o pyproject.toml descreve dependências relevantes, incluindo mlflow, yfinance, xgboost, lightgbm, pandas, numpy, scikit-learn e taskipy.

## Como Executar
1. Executar MLflow UI (em uma thread separada):
   - poetry run mlui
   - ou conforme a configuração do tasks: mlflow ui
2. Rodar o pipeline de treino via Taskipy (configurado no pyproject):
   - poetry run task run
3. Executar o script principal para treinar/avaliar modelos com dados de mercado:
   - poetry run python src/script_mlflow.py

Observação: O script/treino utiliza TimeSeriesSplit para validação de séries temporais e registra métricas no MLflow, incluindo RMSE do conjunto de validação e do conjunto de teste, bem como parâmetros de hiperparametrização.

## Configuração e Hyperparâmetros
- Dados: lag_data(df, lags=30) cria janelas de 30 dias de Close e Volume.
- Predição: default de 5 dias à frente (configurado no pipeline).
- Modelos: XGBoost (xb) e LightGBM (lb) com MultiOutputRegressor para outputs múltiplos.
- Validação: TimeSeriesSplit para evitar leakage em séries temporais.
- Hiperparâmetros logados no MLflow: split_type, model, n_estimators, learning_rate, max_depth, reg_alpha, reg_lambda, etc.

## Estrutura de Dados (Resumo)
- val/DataFrames com features de lag (t0, t1, ..., t{lags-1}) para cada variável de interesse.
- vol_df com variáveis de volume utilizadas como input adicional.

## Boas Práticas para Recrutadores
- O README descreve claramente o objetivo, stack tecnológica, fluxo de treino/validação e como reproduzir o ambiente.
- Inclui instruções de instalação, execução e observação das métricas através do MLflow UI.
- O código está organizado em módulos com responsabilidades bem definidas (coleta de dados, preparação de features e modelagem).

## Contribuições
- Sinta-se à vontade para abrir issues ou pull requests com melhorias na arquitetura de dados, novas estratégias de validação de séries temporais ou experimentos com modelos adicionais.

## Contato
- Criado por Thomás Freire (thomas.fsr@gmail.com)
