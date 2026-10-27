import pandas as pd
import csv
from collections import Counter
import random

random.seed(42)

colunas = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "rotulo"]
df = pd.read_csv("tic-tac-toe.data", header=None, names=colunas)

# print(df.shape)
# print(df["rotulo"].value_counts())
# print(df.isna().sum().sum())
# print(df.duplicated().sum())

linhas = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # linhas
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # colunas
    (0, 4, 8), (2, 4, 6),              # diagonais
]

def classificar(tabuleiro):
    for a, b, c in linhas:
        if tabuleiro[a] == 'x' and tabuleiro[b] == 'x' and tabuleiro[c] == 'x':
            return "X venceu" 
        if tabuleiro[a] == 'o' and tabuleiro[b] == 'o' and tabuleiro[c] == 'o':
            return "O venceu" 
    if 'b' not in tabuleiro:
        return "Empate"
    else:
        return "Tem jogo"
         

with open ("tic-tac-toe.data") as arquivo:
    leitor = csv.DictReader(arquivo, fieldnames = colunas)
    linhas_lidas = list(leitor)

tabuleiros = []

for linha in linhas_lidas:
    tabuleiro = []
    for c in colunas[:9]:
        tabuleiro.append(linha[c])
    tabuleiros.append(tabuleiro)

resultados =[]

for tabuleiro in tabuleiros:
    resultados.append(classificar(tabuleiro))



"""Gerando novos tabuleiros (Tem jogo)"""
def gerar_tabuleiro():
    n = random.randint(1, 8)                  # quantas casas preenchidas
    qtd_x = (n + 1) // 2                      # X começa: X = O ou X = O + 1
    posicoes = random.sample(range(9), n)     # em quais casas

    tabuleiro = ['b'] * 9
    for p in posicoes[:qtd_x]:
        tabuleiro[p] = 'x'
    for p in posicoes[qtd_x:]:
        tabuleiro[p] = 'o'
    return tabuleiro

novos_tabuleiros = []
contagem_por_tamanho = Counter()
LIMITE = 30                                   # máximo por tamanho (8 tamanhos x 30 = 240 > 200)

while len(novos_tabuleiros) < 200:
    t = gerar_tabuleiro()
    tamanho = 9 - t.count('b')

    if (classificar(t) == "Tem jogo"
            and t not in novos_tabuleiros
            and contagem_por_tamanho[tamanho] < LIMITE):
        novos_tabuleiros.append(t)
        contagem_por_tamanho[tamanho] += 1

print(len(novos_tabuleiros))
print(Counter(classificar(t) for t in novos_tabuleiros))
print(sorted(contagem_por_tamanho.items()))


"""Montando o dataset final"""
x_venceu = []
o_venceu = []
empates = []

for i in range(len(tabuleiros)):
    if resultados[i] == "X venceu":
        x_venceu.append(tabuleiros[i])
    elif resultados[i] == "O venceu":
        o_venceu.append(tabuleiros[i])
    else:
        empates.append(tabuleiros[i])

dataset = []

for t in random.sample(x_venceu, 200):
    dataset.append(t + ["X venceu"])
for t in random.sample(o_venceu, 200):
    dataset.append(t + ["O venceu"])
for t in empates:
    dataset.append(t + ["Empate"])
for t in novos_tabuleiros:
    dataset.append(t + ["Tem jogo"])

random.shuffle(dataset)

df_final = pd.DataFrame(dataset, columns=colunas[:9] + ["classe"])
df_final.to_csv("dataset_final.csv", index=False)

print(df_final.shape)                 # esperado: (616, 10)
print(df_final["classe"].value_counts())