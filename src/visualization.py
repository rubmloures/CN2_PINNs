# src/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_wave_propagation(model, config, device, filename="propagacao_onda.png"):
    """
    Gera um heatmap (pcolormesh) da propagação da onda u(x, t).
    """
    model.eval()
    
    # Cria um grid de pontos para avaliação
    x = torch.linspace(config.X_BOUNDS[0], config.X_BOUNDS[1], 200)
    t = torch.linspace(config.T_BOUNDS[0], config.T_BOUNDS[1], 200)
    X, T = torch.meshgrid(x, t, indexing='xy')
    
    # Prepara o input para o modelo
    grid_input = torch.stack((X.flatten(), T.flatten()), dim=1).to(device)
    
    # Faz a predição
    with torch.no_grad():
        u_pred = model(grid_input).reshape(X.shape)
    
    # Converte para numpy para plotar
    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()
    U_np = u_pred.cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T_np, X_np, U_np, cmap='coolwarm', shading='auto', vmin=-1.0, vmax=1.0)
    plt.colorbar(label='Deslocamento u(x,t)')
    plt.xlabel('Tempo (t)')
    plt.ylabel('Posição (x)')
    plt.title(f'Propagação da Onda 1D (Velocidade {config.MODEL_TYPE})')
    
    save_path = os.path.join(config.PLOT_PATH, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot de propagação salvo em: {save_path}")

def plot_wave_snapshots(model, config, device, times=[0.0, 0.25, 0.5, 0.75, 1.0], filename="snapshots_onda.png"):
    """
    Plota "fotos" da onda (u(x) vs x) em diferentes instantes de tempo.
    """
    model.eval()
    x = torch.linspace(config.X_BOUNDS[0], config.X_BOUNDS[1], 500).to(device)
    
    plt.figure(figsize=(10, 6))
    
    for t_val in times:
        if t_val < config.T_BOUNDS[0] or t_val > config.T_BOUNDS[1]:
            continue
            
        t = torch.full_like(x, t_val)
        xt_input = torch.stack((x, t), dim=1)
        
        with torch.no_grad():
            u_pred = model(xt_input)
        
        plt.plot(x.cpu().numpy(), u_pred.cpu().numpy(), label=f't = {t_val:.2f} s')

    plt.xlabel('Posição (x)')
    plt.ylabel('Deslocamento u(x)')
    plt.title(f'Snapshots da Onda (Velocidade {config.MODEL_TYPE})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(config.PLOT_PATH, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot de snapshots salvo em: {save_path}")

def plot_loss_history(history_df, config, filename="loss_history.png"):
    """
    Plota o histórico de todas as componentes da loss.
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