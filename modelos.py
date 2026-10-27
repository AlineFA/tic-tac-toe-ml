# T1 - Tic Tac Toe com ML
# Pré-processamento (A e B), divisão do dataset e 5 classificadores:
# k-NN, Árvore de Decisão, MLP, Regressão Logística e Naive Bayes.
# Entrada: dataset_final.csv (616 linhas: 9 casas + classe)

import time
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report)

SEMENTE = 42  # fixa os sorteios para os resultados saírem iguais a cada execução

# 1) Ler o dataset
df = pd.read_csv("dataset_final.csv")
casas = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
y = df["classe"]  # o que queremos prever

# 2) Pré-processamento

# Abordagem A: só o tabuleiro, em números (x=1, o=-1, vazio=0).
# Usamos -1 para o O (e não 2) para o O não ficar "mais perto" do vazio
# do que o X. Isso importa no k-NN, que usa distância.
mapa = {"x": 1, "o": -1, "b": 0}
XA = df[casas].copy()
for c in casas:
    XA[c] = XA[c].map(mapa)

# Abordagem B: 7 features calculadas a partir do tabuleiro
LINHAS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # linhas
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colunas
    (0, 4, 8), (2, 4, 6),             # diagonais
]

def features_B(t):
    qtd_x = t.count("x")
    qtd_o = t.count("o")
    vazias = t.count("b")
    ocupadas = 9 - vazias

    # Linha com 2 X (ou 2 O) e a terceira casa vazia.
    # Decisão: só conta quando a terceira casa está vazia (anotar no relatório).
    linhas_2x = 0
    linhas_2o = 0
    for a, b, c in LINHAS:
        trio = [t[a], t[b], t[c]]
        if trio.count("x") == 2 and trio.count("b") == 1:
            linhas_2x += 1
        if trio.count("o") == 2 and trio.count("b") == 1:
            linhas_2o += 1

    # Jogador da vez: X começa. Mesma quantidade de X e O -> vez do X (1);
    # X com uma jogada a mais -> vez do O (0).
    vez = 1 if qtd_x == qtd_o else 0

    return [qtd_x, qtd_o, ocupadas, linhas_2x, linhas_2o, vazias, vez]

linhas_B = []
for t in df[casas].values.tolist():
    linhas_B.append(features_B(t))

XB = pd.DataFrame(linhas_B, columns=["qtd_x", "qtd_o", "ocupadas",
                                     "linhas_2x", "linhas_2o", "vazias", "vez"])

abordagens = {"A": XA, "B": XB}

# 3) Divisão do dataset: 60% treino, 20% validação, 20% teste.
# Dividimos os índices uma vez só, então A, B e todos os algoritmos usam
# exatamente as mesmas linhas em cada conjunto.
# stratify mantém a proporção de cada classe (o Empate só tem 16 exemplos).
indices = list(range(len(df)))
idx_treino, idx_temp = train_test_split(
    indices, test_size=0.4, stratify=y, random_state=SEMENTE)
idx_val, idx_teste = train_test_split(
    idx_temp, test_size=0.5, stratify=y.iloc[idx_temp], random_state=SEMENTE)

# Salva a divisão em arquivos (o enunciado pede divisão física)
df.iloc[idx_treino].to_csv("treino.csv", index=False)
df.iloc[idx_val].to_csv("validacao.csv", index=False)
df.iloc[idx_teste].to_csv("teste.csv", index=False)

ytr = y.iloc[idx_treino]
yval = y.iloc[idx_val]
yte = y.iloc[idx_teste]
print("Tamanhos -> treino:", len(idx_treino),
      "| validação:", len(idx_val), "| teste:", len(idx_teste))
print(ytr.value_counts(), "\n")

# 4) Algoritmos e parâmetros a testar (o melhor valor é escolhido na validação)
# Formato: nome: (função que cria o modelo, nome do parâmetro, valores a testar)
algoritmos = {
    # k-NN: vota entre os k tabuleiros mais parecidos (menor distância)
    "k-NN": (lambda v: KNeighborsClassifier(n_neighbors=v),
             "k", [1, 3, 5, 7, 9, 11]),

    # Árvore: faz perguntas do tipo "a casa 4 é x?". max_depth limita a
    # profundidade para evitar overfitting.
    "Árvore de Decisão": (lambda v: DecisionTreeClassifier(max_depth=v, random_state=SEMENTE),
                          "max_depth", [2, 3, 4, 5, 6, 8, None]),

    # MLP: rede neural. (10,) = 1 camada escondida com 10 neurônios;
    # (20, 10) = 2 camadas escondidas.
    "MLP": (lambda v: MLPClassifier(hidden_layer_sizes=v, max_iter=2000,
                                    random_state=SEMENTE),
            "camadas_escondidas", [(5,), (10,), (20,), (10, 5), (20, 10)]),

    # Regressão Logística: modelo linear que calcula uma probabilidade
    # para cada classe. C controla a regularização (menor = modelo mais simples).
    "Regressão Logística": (lambda v: LogisticRegression(C=v, max_iter=2000),
                            "C", [0.01, 0.1, 1, 10, 100]),

    # Naive Bayes: calcula a probabilidade de cada classe assumindo que as
    # features são independentes. var_smoothing evita variância zero.
    "Naive Bayes": (lambda v: GaussianNB(var_smoothing=v),
                    "var_smoothing", [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]),
}

def metricas(y_real, y_pred):
    # Média macro: todas as classes pesam igual (importante pro Empate, que é pequeno)
    return {
        "acuracia": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_real, y_pred, average="macro", zero_division=0),
    }

# 5) Treinar, escolher parâmetros na validação e avaliar no teste
resultados = []
objetos = {}  # guarda modelo e scaler de cada combinação

for nome_abord, X in abordagens.items():

    # Normalização (média 0, desvio 1). O scaler aprende só com o treino,
    # para não vazar informação da validação/teste.
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X.iloc[idx_treino])
    Xval = scaler.transform(X.iloc[idx_val])
    Xte = scaler.transform(X.iloc[idx_teste])

    for nome_alg, (criar, nome_param, valores) in algoritmos.items():

        # (a) testa cada valor do parâmetro e vê qual vai melhor na validação
        melhor_valor = None
        melhor_f1_val = -1
        for v in valores:
            modelo = criar(v)
            modelo.fit(Xtr, ytr)
            f1_val = f1_score(yval, modelo.predict(Xval), average="macro", zero_division=0)
            print(f"[{nome_abord}] {nome_alg} {nome_param}={v}  F1 validação = {f1_val:.3f}")
            if f1_val > melhor_f1_val:
                melhor_f1_val = f1_val
                melhor_valor = v

        # (b) retreina com o melhor parâmetro, medindo o tempo de treino
        modelo = criar(melhor_valor)
        t0 = time.perf_counter()
        modelo.fit(Xtr, ytr)
        tempo_treino = time.perf_counter() - t0

        # (c) avalia no teste (só agora o teste é usado)
        t0 = time.perf_counter()
        pred_teste = modelo.predict(Xte)
        tempo_pred = time.perf_counter() - t0

        m = metricas(yte, pred_teste)
        acc_treino = accuracy_score(ytr, modelo.predict(Xtr))  # para checar overfitting

        resultados.append({
            "abordagem": nome_abord,
            "algoritmo": nome_alg,
            "parametro": f"{nome_param}={melhor_valor}",
            "f1_validacao": melhor_f1_val,
            "acc_treino": acc_treino,
            "acc_teste": m["acuracia"],
            "precision_teste": m["precision"],
            "recall_teste": m["recall"],
            "f1_teste": m["f1"],
            "tempo_treino_s": tempo_treino,
            "tempo_pred_s": tempo_pred,
        })
        objetos[(nome_abord, nome_alg)] = (modelo, scaler)

        print(f"\n>>> [{nome_abord}] {nome_alg} ({nome_param}={melhor_valor}) no TESTE:")
        print(classification_report(yte, pred_teste, zero_division=0))

# 6) Comparação: tabela e gráficos
res = pd.DataFrame(resultados)
res.to_csv("resultados.csv", index=False)
print("\nTabela final")
print(res.round(3).to_string(index=False))

# Gráfico: F1 no teste por algoritmo, comparando as abordagens A e B
pivo = res.pivot(index="algoritmo", columns="abordagem", values="f1_teste")
pivo.plot(kind="bar", ylim=(0, 1.05))
plt.ylabel("F1 (macro) no teste")
plt.title("Abordagem A x B por algoritmo")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("comparacao_f1.png", dpi=150)

# Gráfico: tempo de treino
pivo_t = res.pivot(index="algoritmo", columns="abordagem", values="tempo_treino_s")
pivo_t.plot(kind="bar")
plt.ylabel("Tempo de treino (s)")
plt.title("Custo de treino por algoritmo e abordagem")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("comparacao_tempo.png", dpi=150)

# 7) Salvar o melhor modelo (escolhido pela validação, não pelo teste)
melhor = res.sort_values("f1_validacao", ascending=False).iloc[0]
modelo_f, scaler_f = objetos[(melhor["abordagem"], melhor["algoritmo"])]
joblib.dump({"modelo": modelo_f, "scaler": scaler_f, "abordagem": melhor["abordagem"]},
            "melhor_modelo.joblib")
print(f"\nMelhor (validação): {melhor['algoritmo']} com abordagem {melhor['abordagem']} "
      f"({melhor['parametro']}) -> salvo em melhor_modelo.joblib")