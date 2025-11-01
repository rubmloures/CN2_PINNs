# main_2d.py
import os
import sys

# Adiciona o diretório 'src_2' ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src_2')))

try:
    # Importa os módulos do pacote src_2 
    from src_2.trainer import run_training
    from src_2.utils import setup_device
    from src_2.visualization import plot_wave_snapshots_2d, plot_loss_history
    import config.config_2d_variavel as config_2d
except ImportError as e:
    print("Erro ao importar módulos; execute este script a partir da raiz do repositório ou ajuste PYTHONPATH.")
    print(f"Detalhe do erro: {e}")
    sys.exit(1)

def main():
    """
    Função principal para treinar o modelo 2D.
    """
    config = config_2d

    print(f"--- Iniciando Experimento: {config.MODEL_TYPE} ---")
    print(config.DESCRIPTION)

    # Garante que as pastas de resultados existam
    os.makedirs(config.PLOT_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    # 1. Treinamento
    model, history_df = run_training(config)
    
    # 2. Visualização
    print("Treinamento concluído. Gerando plots...")
    device = setup_device(config)
    
    model.eval() 

    # Gera os plots 2D
    plot_wave_snapshots_2d(model, config, device, 
                           times=[0.0, 0.25, 0.5, 0.75], 
                           filename="snapshots_2d_final.png")
    plot_loss_history(history_df, config, filename="loss_history_2d_final.png")
    
    print(f"--- Experimento {config.MODEL_TYPE} concluído ---")
    print(f"Modelo salvo em: {config.MODEL_PATH}")
    print(f"Plots salvos em: {config.PLOT_PATH}")

if __name__ == "__main__":
    main()