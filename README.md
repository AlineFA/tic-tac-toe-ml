# Tic Tac Toe com Machine Learning

Trabalho T1 da disciplina de **Inteligência Artificial** (PUCRS), por Aline Fassina Acevedo.

O projeto treina modelos de ML para **classificar tabuleiros de jogo da velha** em quatro classes: `X venceu`, `O venceu`, `Empate` e `Tem jogo` (partida ainda em andamento). Depois, o melhor modelo é usado em um jogo com interface gráfica, onde um humano joga contra uma máquina aleatória e a IA tenta dizer, a cada jogada, em que estado o jogo está.

🎥 Vídeo de apresentação: https://youtu.be/kl_IUC5lSCc?si=jdyyC9pEXSg2gNg5

---

## Como funciona

**1. Dataset** (`tratamento_dataset.py`)
O dataset original da UCI (*Tic-Tac-Toe Endgame*) só tem tabuleiros finais. Eu reclassifiquei os tabuleiros em 4 classes pela regra do jogo e gerei 200 tabuleiros novos de "Tem jogo" (partidas em andamento, sem duplicatas e com no máximo 30 por quantidade de casas preenchidas). O dataset final tem **616 tabuleiros**: 200 de X venceu, 200 de O venceu, 16 de Empate e 200 de Tem jogo.

**2. Duas abordagens de pré-processamento** (`modelos.py`)
- **A:** só o tabuleiro em números (x = 1, o = -1, vazio = 0). Usei -1 para o O, e não 2, para que o O não fique "mais perto" do vazio do que o X (isso importa no k-NN, que usa distância).
- **B:** 7 features calculadas a partir do tabuleiro: quantidade de X, de O, casas ocupadas, casas vazias, linhas com 2 X e a terceira casa vazia, linhas com 2 O e a terceira vazia, e de quem é a vez.

**3. Divisão e treino**
Divisão estratificada em 60% treino, 20% validação e 20% teste (salvas em `treino.csv`, `validacao.csv` e `teste.csv`). A normalização aprende só com o treino, para não vazar informação. Cada algoritmo testa vários valores de parâmetro, e o melhor é escolhido na **validação**; o teste só é usado no final.

**4. Algoritmos:** k-NN, Árvore de Decisão, MLP, Regressão Logística e Naive Bayes, cada um nas abordagens A e B (10 modelos).

**5. Métricas:** acurácia, precision, recall e F1 com média *macro* (todas as classes pesam igual, importante porque Empate tem só 16 exemplos).

## Resultados

Melhores resultados no conjunto de teste (tabela completa em `resultados.csv`):

| Abordagem | Algoritmo | Parâmetro | Acurácia | F1 macro |
|---|---|---|---|---|
| B | **k-NN** | k = 1 | 0,887 | **0,865** |
| B | Naive Bayes | var_smoothing = 0,1 | 0,839 | 0,825 |
| A | MLP | 1 camada de 20 neurônios | 0,879 | 0,670 |
| B | Regressão Logística | C = 0,1 | 0,871 | 0,661 |
| A | Regressão Logística | C = 1 | 0,573 | 0,396 |

O **k-NN com a abordagem B** teve o melhor F1 na validação e foi salvo como `melhor_modelo.joblib`. As features calculadas (abordagem B) melhoraram o F1 em 4 dos 5 algoritmos em relação ao tabuleiro bruto (o MLP ficou praticamente igual). Os gráficos estão em `comparacao_f1.png` e `comparacao_tempo.png`.

## Front-end

`frontend.py` abre uma janela (tkinter): você joga de X e a máquina joga de O em casas aleatórias. A cada jogada, a IA classifica o tabuleiro e o placar conta os acertos e erros dela em relação à regra real do jogo.

## Como executar

Requisitos: Python 3 com `pandas`, `scikit-learn`, `joblib` e `matplotlib` (o `tkinter` já vem com o Python).

```bash
pip install pandas scikit-learn joblib matplotlib

python frontend.py            # jogar contra a máquina (usa melhor_modelo.joblib)
python modelos.py             # refazer treino e avaliação (opcional)
python tratamento_dataset.py  # refazer o dataset (opcional; reescreve dataset_final.csv)
```

## Arquivos

| Arquivo | O que é |
|---|---|
| `tic-tac-toe.data` | dataset original da UCI |
| `tratamento_dataset.py` | reclassifica o dataset em 4 classes e gera os tabuleiros "Tem jogo" |
| `dataset_final.csv` | dataset final (616 tabuleiros) |
| `modelos.py` | pré-processamento, divisão, treino e avaliação dos 5 algoritmos; salva o melhor modelo |
| `treino.csv`, `validacao.csv`, `teste.csv` | divisão do dataset (60/20/20) |
| `resultados.csv` | métricas dos 10 modelos |
| `comparacao_f1.png`, `comparacao_tempo.png` | gráficos de F1 e de tempo de treino |
| `melhor_modelo.joblib` | melhor modelo treinado (k-NN, abordagem B, k = 1) |
| `frontend.py` | jogo: humano contra máquina aleatória, com a IA classificando o tabuleiro |
