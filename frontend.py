# T1 - Tic Tac Toe com ML: front end com janela (tkinter)
# Humano joga de X (começa), a máquina joga de O em casas aleatórias.
# A cada jogada a IA classifica o tabuleiro e o placar conta acertos e erros.
#
# Regras de fim de jogo:
#  - se o jogo realmente terminou (alguém ganhou ou deu empate), a partida acaba,
#    mesmo que a IA não tenha percebido (conta como erro da IA);
#  - se a IA acha que terminou mas o jogo ainda não terminou, a partida continua
#    (conta como erro da IA).
#
# Como rodar:  python frontend.py
# Precisa estar na mesma pasta que o melhor_modelo.joblib

import random
import tkinter as tk

import joblib
import pandas as pd

LINHAS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]
MAPA = {"x": 1, "o": -1, "b": 0}
CASAS = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
COLUNAS_B = ["qtd_x", "qtd_o", "ocupadas", "linhas_2x", "linhas_2o", "vazias", "vez"]


def features_B(t):
    # Tem que ser IGUAL à função usada no modelos.py
    qtd_x = t.count("x")
    qtd_o = t.count("o")
    vazias = t.count("b")
    ocupadas = 9 - vazias
    linhas_2x = 0
    linhas_2o = 0
    for a, b, c in LINHAS:
        trio = [t[a], t[b], t[c]]
        if trio.count("x") == 2 and trio.count("b") == 1:
            linhas_2x += 1
        if trio.count("o") == 2 and trio.count("b") == 1:
            linhas_2o += 1
    vez = 1 if qtd_x == qtd_o else 0
    return [qtd_x, qtd_o, ocupadas, linhas_2x, linhas_2o, vazias, vez]


def classificar(t):
    # Resposta CERTA (regra do jogo), usada para conferir a IA
    for a, b, c in LINHAS:
        if t[a] == t[b] == t[c] and t[a] != "b":
            return "X venceu" if t[a] == "x" else "O venceu"
    if "b" not in t:
        return "Empate"
    return "Tem jogo"


def prever(pacote, t):
    # Resposta da IA
    if pacote["abordagem"] == "A":
        entrada = pd.DataFrame([[MAPA[v] for v in t]], columns=CASAS)
    else:
        entrada = pd.DataFrame([features_B(t)], columns=COLUNAS_B)
    entrada = pacote["scaler"].transform(entrada)
    return pacote["modelo"].predict(entrada)[0]


class Jogo:
    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("Jogo da Velha com IA")
        self.pacote = joblib.load("melhor_modelo.joblib")
        self.acertos = 0
        self.erros = 0

        tk.Label(raiz, text="Você é o X. A máquina é o O.",
                 font=("Helvetica", 14)).pack(pady=(10, 4))

        quadro = tk.Frame(raiz)
        quadro.pack(padx=10, pady=4)
        self.botoes = []
        for i in range(9):
            b = tk.Button(quadro, text="", width=4, height=2,
                          font=("Helvetica", 32, "bold"),
                          command=lambda i=i: self.clicar(i))
            b.grid(row=i // 3, column=i % 3, padx=2, pady=2)
            self.botoes.append(b)

        self.msg = tk.Label(raiz, text="", font=("Helvetica", 13),
                            wraplength=360, justify="center")
        self.msg.pack(pady=6)
        self.placar = tk.Label(raiz, text="", font=("Helvetica", 13, "bold"))
        self.placar.pack()

        linha = tk.Frame(raiz)
        linha.pack(pady=10)
        tk.Button(linha, text="Novo jogo", command=self.novo_jogo).grid(row=0, column=0, padx=6)
        tk.Button(linha, text="Zerar placar", command=self.zerar_placar).grid(row=0, column=1, padx=6)

        self.novo_jogo()

    def novo_jogo(self):
        self.tab = ["b"] * 9
        self.fim = False
        self.aguardando = False
        for b in self.botoes:
            b.config(text="", state="normal")
        self.msg.config(text="Sua vez! Clique em uma casa.")
        self.atualizar_placar()

    def zerar_placar(self):
        self.acertos = 0
        self.erros = 0
        self.atualizar_placar()

    def atualizar_placar(self):
        total = self.acertos + self.erros
        pct = f"{100 * self.acertos / total:.1f}%" if total else "-"
        self.placar.config(
            text=f"IA: {self.acertos} acertos | {self.erros} erros | {pct}")

    def clicar(self, i):
        if self.fim or self.aguardando or self.tab[i] != "b":
            return
        self.tab[i] = "x"
        self.botoes[i].config(text="X")
        self.avaliar()
        if not self.fim:
            self.aguardando = True
            self.raiz.after(400, self.jogada_maquina)

    def jogada_maquina(self):
        vazias = [i for i in range(9) if self.tab[i] == "b"]
        j = random.choice(vazias)
        self.tab[j] = "o"
        self.botoes[j].config(text="O")
        self.aguardando = False
        self.avaliar()

    def avaliar(self):
        real = classificar(self.tab)
        ia = prever(self.pacote, self.tab)

        if ia == real:
            self.acertos += 1
            texto = f"IA: \"{ia}\" (acertou)"
        else:
            self.erros += 1
            texto = f"IA: \"{ia}\" (errou, o certo era \"{real}\")"

        if real != "Tem jogo":
            self.fim = True
            for b in self.botoes:
                b.config(state="disabled")
            texto += f"\nFim de jogo: {real}. Clique em Novo jogo."
        elif ia != "Tem jogo":
            texto += "\nA IA achou que acabou, mas o jogo continua."
        else:
            texto += "\nO jogo continua."

        self.msg.config(text=texto)
        self.atualizar_placar()


if __name__ == "__main__":
    raiz = tk.Tk()
    Jogo(raiz)
    raiz.mainloop()