# src_2/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def plot_wave_snapshots_2d(model, config, device, times=[0.0, 0.25, 0.5, 0.75], filename="snapshots_2d.png"):
    """
    Plota "fotos" da onda 2D (u(x,y) vs x,y) em diferentes instantes de tempo.
    """
    model.eval()
    
    # Cria um grid 2D para avaliação (tenta obter resoluções do config)
    x_res = getattr(config, 'N_X', None) or getattr(config, 'GRID_NX', None) or 100
    y_res = getattr(config, 'N_Y', None) or getattr(config, 'GRID_NY', None) or x_res
    x = torch.linspace(config.X_BOUNDS[0], config.X_BOUNDS[1], int(x_res), device=device)
    y = torch.linspace(config.Y_BOUNDS[0], config.Y_BOUNDS[1], int(y_res), device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    # Avalia o modelo em cada tempo e guarda resultados (autoscaling)
    n_times = len(times)
    n_cols = 2
    n_rows = math.ceil(n_times / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    snapshots = []
    vmin, vmax = float('inf'), float('-inf')

    for i, t_val in enumerate(times):
        if t_val < config.T_BOUNDS[0] or t_val > config.T_BOUNDS[1]:
            axes[i].set_title(f"Tempo t={t_val:.2f} s (fora dos limites)")
            axes[i].axis('off')
            continue
        T = torch.full_like(X, float(t_val), device=device)
        # Prepara o input (x, y, t)
        grid_input = torch.stack((X.flatten(), Y.flatten(), T.flatten()), dim=1).to(device)

        # Faz a predição
        with torch.no_grad():
            u_pred = model(grid_input).cpu().numpy().reshape(X.cpu().numpy().shape)

        # guarde também o índice do eixo para manter o mapeamento correto
        snapshots.append((i, u_pred))
        vmin = min(vmin, float(np.min(u_pred)))
        vmax = max(vmax, float(np.max(u_pred)))

        # Prepara o subplot (o heatmap será desenhado depois com vmin/vmax globais)
        ax = axes[i]
        ax.set_xlabel('Posição (x)')
        ax.set_ylabel('Posição (y)')
        ax.set_title(f'Snapshot da Onda 2D em t = {t_val:.2f} s')
        ax.set_aspect('equal', 'box')

    # Desenha os snapshots agora com escala global
    cax = None
    # snapshots contém tuplas (axis_index, u_array)
    for axis_idx, u in snapshots:
        ax = axes[axis_idx]
        cax = ax.pcolormesh(X_np, Y_np, u, cmap='seismic', shading='auto', vmin=vmin, vmax=vmax)

    # Remove eixos extras
    for i in range(n_times, len(axes)):
        fig.delaxes(axes[i])

    # Adiciona a barra de cores apenas se houver pelo menos um snapshot
    if cax is not None:
        fig.colorbar(cax, ax=axes.tolist(), orientation='vertical', fraction=0.02, pad=0.04, label='Deslocamento u(x,y,t)')

    plt.tight_layout()
    save_path = os.path.join(config.PLOT_PATH, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot de snapshots 2D salvo em: {save_path}")


def plot_loss_history(history_df, config, filename="loss_history.png"):
    """
    Plota o histórico de todas as componentes da loss. (Idêntico ao 1D)
    """
    plt.figure(figsize=(12, 8))
    
    plt.semilogy(history_df['Epoch'], history_df['Total Loss'], label='Total Loss')
    plt.semilogy(history_df['Epoch'], history_df['PDE Loss'], label='PDE Loss', alpha=0.7)
    plt.semilogy(history_df['Epoch'], history_df['IC Loss'], label='IC Loss', alpha=0.7)
    plt.semilogy(history_df['Epoch'], history_df['BC Loss'], label='BC Loss', alpha=0.7)
    
    plt.title('Histórico de Loss Durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    save_path = os.path.join(config.PLOT_PATH, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot de loss salvo em: {save_path}")