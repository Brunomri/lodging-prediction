# Case 2 – Regressão de Preços de Hospedagem

Este é o segundo projeto prático do AI Bootcamp. O objetivo é aprofundar conceitos de Machine Learning aplicados a um problema de regressão, como viés-variância, regularização, engenharia de features, tratamento de variáveis categóricas, avaliação de modelos, e divisão entre treino e teste.

## Descrição do Problema

Você faz parte da equipe de Data Product Management de uma start-up de aluguel de casas na Europa. Com o crescimento da empresa, é necessário desenvolver um modelo de regressão para prever o valor esperado do aluguel com base em informações da acomodação, como:

- Número de hóspedes
- Quartos
- Distância ao centro
- Latitude e longitude
- Dia da semana

A base de dados inclui 10 cidades: Amsterdã, Atenas, Barcelona, Berlim, Budapeste, Lisboa, Londres, Paris, Roma e Viena. Escolha 3 cidades para realizar a análise. Os dados estão separados por cidade e por dias úteis/fins de semana.

## Objetivo

Explorar os dados, identificar relações entre variáveis e construir um modelo de regressão único para prever preços. O modelo será avaliado com MAE (Mean Absolute Error). Também é necessário apresentar visualmente as características mais relevantes do modelo.

## Dicas

- Use One-Hot Encoding para variáveis categóricas.
- Separe dados em treino e teste.
- Crie novas features combinando variáveis.
- Estude underfitting, overfitting e o trade-off viés-variância.
- Utilize regularizações L1 (Lasso) e L2 (Ridge).
- Normalize features quando necessário.
- Avalie modelos como Regressão Linear, Lasso, Ridge, ElasticNet, Random Forest, XGBoost, LightGBM, CatBoost, SVR.