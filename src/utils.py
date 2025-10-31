# src/utils.py
import torch
import numpy as np
import random
import os
import pandas as pd

def set_seed(seed):
    """Define a seed para reprodutibilidade."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def setup_device(config):
    """Configura o dispositivo (CPU ou CUDA)."""
    if "cuda" in config.DEVICE and not torch.cuda.is_available():
        print("CUDA não disponível. Usando CPU.")
        return torch.device("cpu")
    return torch.device(config.DEVICE)

def save_model(model, config):
    """Salva os pesos do modelo."""
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)

def load_model(model_class, config, device):
    """Carrega um modelo treinado."""
    model = model_class(config.LAYERS).to(device)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        model.eval()
        print(f"Modelo carregado de {config.MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"Erro: Arquivo do modelo não encontrado em {config.MODEL_PATH}")
        return None

def save_training_history(history_df, config):
    """Salva o histórico de treinamento em um CSV."""
    os.makedirs(os.path.dirname(config.HISTORY_PATH), exist_ok=True)
    history_df.to_csv(config.HISTORY_PATH, index=False)
    print(f"Histórico de treinamento salvo em {config.HISTORY_PATH}")