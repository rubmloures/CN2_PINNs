# main.py
import argparse
import os
import sys

# Adiciona o diretório 'src' ao path para importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.trainer import run_training
    from src.utils import setup_device
    from src.visualization import plot_wave_propagation, plot_wave_snapshots, plot_loss_history
    import config.config_constante as config_const
    import config.config_variavel as config_var
except ImportError as e:
    print(f"Erro ao importar módulos. Verifique se a estrutura de pastas está correta.")
    print(f"Detalhe do erro: {e}")
    sys.exit(1)

def main(model_type):
    """
    Função principal para carregar a configuração, treinar e gerar visualizações.
    """
    if model_type == 'constante':
        config = config_const
    elif model_type == 'variavel':
        config = config_var
    else:
        print(f"Tipo de modelo '{model_type}' não reconhecido. Use 'constante' ou 'variavel'.")
        return

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
    
    # Coloca o modelo em modo de avaliação (importante para dropout/batchnorm, se houver)
    model.eval() 

    # Gera os plots principais para sua apresentação
    plot_wave_propagation(model, config, device, filename="propagacao_onda_final.png")
    plot_wave_snapshots(model, config, device, filename="snapshots_onda_final.png")
    plot_loss_history(history_df, config, filename="loss_history_final.png")
    
    print(f"--- Experimento {config.MODEL_TYPE} concluído ---")
    print(f"Modelo salvo em: {config.MODEL_PATH}")
    print(f"Plots salvos em: {config.PLOT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar PINN para Equação da Onda 1D.")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['constante', 'variavel'],
        help="Tipo de modelo a ser treinado (velocidade constante ou variável)."
    )
    args = parser.parse_args()
    main(args.model)