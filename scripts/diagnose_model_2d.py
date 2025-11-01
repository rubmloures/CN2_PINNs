"""Diagnóstico rápido: carrega o modelo salvo (se existir) e avalia min/max/mean em alguns tempos."""
import os
import sys
import torch
import numpy as np

# Ajusta path para permitir imports de src_2 e config
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from src_2.utils import setup_device, load_model
import config.config_2d_variavel as cfg
from src_2.model import PINN


def main():
    device = setup_device(cfg)
    print(f"Usando device: {device}")

    model = load_model(PINN, cfg, device)
    if model is None:
        print("Modelo não pôde ser carregado. Verifique config.MODEL_PATH e a arquitetura.")
        return

    model.eval()

    # Grid coarse para teste
    x = torch.linspace(cfg.X_BOUNDS[0], cfg.X_BOUNDS[1], 100, device=device)
    y = torch.linspace(cfg.Y_BOUNDS[0], cfg.Y_BOUNDS[1], 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    times = [0.0, 0.25 * (cfg.T_BOUNDS[1] - cfg.T_BOUNDS[0]) + cfg.T_BOUNDS[0],
             0.5 * (cfg.T_BOUNDS[1] - cfg.T_BOUNDS[0]) + cfg.T_BOUNDS[0]]

    for t_val in times:
        T = torch.full_like(X, float(t_val), device=device)
        pts = torch.stack((X.flatten(), Y.flatten(), T.flatten()), dim=1)
        with torch.no_grad():
            try:
                u = model(pts).cpu().numpy().reshape(X.cpu().numpy().shape)
            except Exception as e:
                print(f"Erro ao avaliar o modelo em t={t_val}: {e}")
                continue
        print(f"t={t_val:.3f} -> min={u.min():.3e}, max={u.max():.3e}, mean={u.mean():.3e}")

if __name__ == '__main__':
    main()
