# config/config_variavel.py

import torch

# --- Identificação do Modelo ---
MODEL_TYPE = "variavel"
DESCRIPTION = "Simulação 1D da Equação da Onda com velocidade variável c(x) = 1.0 + 0.5*x"

# --- Domínio Espaço-Temporal ---
X_BOUNDS = [0.0, 1.0]  # Limites espaciais (x)
T_BOUNDS = [0.0, 1.0]  # Limites temporais (t)

# --- Parâmetros Físicos ---
# Definimos c(x) = C_BASE + C_GRAD * x
C_BASE = 1.0
C_GRAD = 0.5

# --- Parâmetros de Treinamento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
EPOCHS = 30000  # Pode precisar de mais épocas para convergir
# Número de pontos amostrados a cada época
N_IC = 200
N_BC = 200
N_PDE = 15000 # Mais pontos de PDE podem ajudar

# --- Arquitetura da Rede ---
LAYERS = [2, 32, 32, 32, 32, 1]

# --- Pesos da Loss Function ---
W_PDE = 1.0
W_IC_U = 1.0
W_IC_V = 0.1
W_BC = 1.0

# --- Caminhos de Saída ---
SAVE_PATH = "resultados/variavel/"
MODEL_PATH = "resultados/variavel/modelo/best_model.pth"
PLOT_PATH = "resultados/variavel/plots/"
HISTORY_PATH = "resultados/variavel/training_history.csv"