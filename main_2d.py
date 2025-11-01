# main_2d.py
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src_2')))

try:
    from trainer import run_training
    from utils import setup_device, load_model
    from visualization import plot_wave_snapshots_2d, plot_loss_history, plot_wave_surface_3d
    import config.config_2d_variavel as config_2d
    from model import PINN # Precisa importar a classe PINN para load_model
except ImportError as e:
    print(f"Erro ao importar módulos do 'src_2'. Verifique a estrutura de pastas.")
    print(f"Detalhe do erro: {e}")
    sys.exit(1)

def main():
    """
    Função principal para treinar o modelo 2D.
    """
    config = config_2d

    print(f"--- Iniciando Experimento: {config.MODEL_TYPE} ---")
    print(config.DESCRIPTION)

    os.makedirs(config.PLOT_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    # 1. Treinamento
    #model, history_df = run_training(config) # Comente ou remova se já treinou e quer só gerar plots
    
    # Se você já treinou, pode carregar o modelo em vez de treinar novamente:
    device = setup_device(config)
    model = load_model(PINN, config, device) # Carrega o modelo
    
    if model is None: # Se o carregamento falhar (e.g., primeira vez rodando), treina.
        print("Modelo não encontrado ou carregado com erro. Iniciando treinamento...")
        model, history_df = run_training(config)
    else: # Se carregou com sucesso, apenas carrega o histórico para plotar
        history_df = pd.read_csv(config.HISTORY_PATH)


    # 2. Visualização
    print("Treinamento concluído. Gerando plots...")
    
    model.eval() 

    plot_wave_snapshots_2d(model, config, device, 
                           times=[0.0, 0.25, 0.5, 0.75], 
                           filename="snapshots_2d_final.png")
    
    plot_wave_surface_3d(model, config, device, t_val=0.0, filename="surface_3d_t0.00.png")
    plot_wave_surface_3d(model, config, device, t_val=0.25, filename="surface_3d_t0.25.png")
    plot_wave_surface_3d(model, config, device, t_val=0.50, filename="surface_3d_t0.50.png")
    plot_wave_surface_3d(model, config, device, t_val=0.75, filename="surface_3d_t0.75.png")

    plot_loss_history(history_df, config, filename="loss_history_2d_final.png")
    
    print(f"--- Experimento {config.MODEL_TYPE} concluído ---")
    print(f"Modelo salvo em: {config.MODEL_PATH}")
    print(f"Plots salvos em: {config.PLOT_PATH}")

if __name__ == "__main__":
    main()