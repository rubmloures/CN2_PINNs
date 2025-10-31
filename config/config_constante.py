# config/config_constante.py

import torch

# --- Identificação do Modelo ---
MODEL_TYPE = "constante"
DESCRIPTION = "Simulação 1D da Equação da Onda com velocidade constante c=1.0"

# --- Domínio Espaço-Temporal ---
X_BOUNDS = [0.0, 1.0]  # Limites espaciais (x)
T_BOUNDS = [0.0, 1.0]  # Limites temporais (t)

# --- Parâmetros Físicos ---
# Velocidade da onda
C = 1.0

# --- Parâmetros de Treinamento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
EPOCHS = 20000
# Número de pontos amostrados a cada época
N_IC = 200  # Pontos de Condição Inicial (t=0)
N_BC = 200  # Pontos de Condição de Contorno (x=0, x=L)
N_PDE = 10000 # Pontos de Colocação (resíduo da PDE)

# --- Arquitetura da Rede ---
# [input_dim, hidden_1, hidden_2, ..., output_dim]
# Input: (x, t) -> 2 neurônios
# Output: u(x, t) -> 1 neurônio
LAYERS = [2, 32, 32, 32, 32, 1]

# --- Pesos da Loss Function ---
W_PDE = 1.0       # Peso para o resíduo da PDE
W_IC_U = 1.0      # Peso para a condição inicial u(x,0)
W_IC_V = 0.1      # Peso para a condição inicial u_t(x,0) (velocidade)
W_BC = 1.0       # Peso para as condições de contorno

# --- Caminhos de Saída ---
SAVE_PATH = "resultados/constante/"
MODEL_PATH = "resultados/constante/modelo/best_model.pth"
PLOT_PATH = "resultados/constante/plots/"
HISTORY_PATH = "resultados/constante/training_history.csv"