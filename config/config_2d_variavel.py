# config/config_2d_variavel.py

import torch

# --- Identificação do Modelo ---
MODEL_TYPE = "simulacao_2d"
DESCRIPTION = "Simulação 2D da Equação da Onda com velocidade variável c(x, y)"

# --- Domínio Espaço-Temporal ---
X_BOUNDS = [0.0, 1.0]  # Limites espaciais (x)
Y_BOUNDS = [0.0, 1.0]  # Limites espaciais (y)
T_BOUNDS = [0.0, 1.0]  # Limites temporais (t)

# --- Parâmetros Físicos ---
# Velocidade variável c(x, y) = C_BASE + C_GRAD_X * x + C_GRAD_Y * y
C_BASE = 1.0
C_GRAD_X = 0.5
C_GRAD_Y = 0.3

# --- Parâmetros de Treinamento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
EPOCHS = 30000 # Problemas 2D são mais difíceis, podem precisar de mais
# Número de pontos (aumentado para o domínio 3D)
N_IC = 1000   # Pontos de Condição Inicial (t=0)
N_BC = 1000   # Pontos de Condição de Contorno (em cada uma das 4 bordas)
N_PDE = 20000 # Pontos de Colocação (resíduo da PDE)

# --- Arquitetura da Rede ---
# [input_dim, hidden_1, ..., output_dim]
# Input: (x, y, t) -> 3 neurônios
# Output: u(x, y, t) -> 1 neurônio
LAYERS = [3, 40, 40, 40, 40, 1]

# --- Pesos da Loss Function ---
W_PDE = 1.0
W_IC_U = 1.0
W_IC_V = 0.1
W_BC = 1.0

# --- Caminhos de Saída ---
SAVE_PATH = "resultados/simulacao_2d/"
MODEL_PATH = "resultados/simulacao_2d/modelo/best_model.pth"
PLOT_PATH = "resultados/simulacao_2d/plots/"
HISTORY_PATH = "resultados/simulacao_2d/training_history.csv"