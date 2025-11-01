"""Testa a chamada de plot_wave_snapshots_2d carregando o modelo salvo e gerando o PNG (sem treinar).
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from src_2.utils import setup_device, load_model
from src_2.visualization import plot_wave_snapshots_2d
from src_2.model import PINN
import config.config_2d_variavel as cfg


def main():
    device = setup_device(cfg)
    model = load_model(PINN, cfg, device)
    if model is None:
        print("Modelo não carregado; abortando teste de plotagem.")
        return
    plot_wave_snapshots_2d(model, cfg, device, times=[0.0, 0.25, 0.5, 0.75], filename="snapshots_2d_test.png")

if __name__ == '__main__':
    main()
