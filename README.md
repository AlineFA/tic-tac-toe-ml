T1 - TIC TAC TOE COM ML
Inteligencia Artificial 
Aline Acevedo

Video: https://youtu.be/kl_IUC5lSCc?si=jdyyC9pEXSg2gNg5


ARQUIVOS

tic-tac-toe.data         dataset original da UCI
tratamento_dataset.py    reclassifica o dataset em 4 classes e gera os
                         tabuleiros "Tem jogo"
dataset_final.csv        dataset final (616 tabuleiros)
modelos.py               pre-processamento, divisao, treino e avaliacao dos
                         5 algoritmos; salva o melhor modelo
treino.csv               divisao do dataset (60% treino)
validacao.csv            divisao do dataset (20% validacao)
teste.csv                divisao do dataset (20% teste)
resultados.csv           metricas dos 10 modelos
comparacao_f1.png        grafico de F1
comparacao_tempo.png     grafico de tempo de treino
melhor_modelo.joblib     melhor modelo treinado (k-NN, abordagem B, k=1)
frontend.py              front end: humano contra maquina aleatoria


COMO EXECUTAR

Requisitos: Python 3 com pandas, scikit-learn, joblib e matplotlib
(o tkinter ja vem com o Python).

Abra o terminal na pasta do trabalho e rode:

1) Jogar (front end):

  python frontend.py

   Se der erro ao carregar o modelo, rode antes o modelos.py (passo 2).

2) Refazer o treino e a avaliacao (opcional):

  python modelos.py

3) Refazer o dataset (opcional; reescreve o dataset_final.csv):

  python tratamento_dataset.py