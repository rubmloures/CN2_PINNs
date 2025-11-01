"""Testa o plot de superfície 3D da onda carregando o modelo salvo.
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from src_2.utils import setup_device, load_model
from src_2.visualization import plot_wave_surface_3d
from src_2.model import PINN
import config.config_2d_variavel as cfg


def main():
    device = setup_device(cfg)
    model = load_model(PINN, cfg, device)
    if model is None:
        print("Modelo não carregado; abortando teste de plotagem.")
        return

    # Plota superfície 3D em alguns tempos
    times = [0.0, 0.25, 0.5, 0.75]
    for t in times:
        plot_wave_surface_3d(model, cfg, device, t,
                            filename=f"wave_surface_3d_t{t:.2f}.png")

if __name__ == '__main__':
    main()