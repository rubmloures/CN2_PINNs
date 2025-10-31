# src/trainer.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm

from src.model import PINN
from src.data_loader import get_training_data
from src.physics import compute_pde_residual, compute_ic_derivatives
from src.utils import set_seed, setup_device, save_model, save_training_history

def compute_loss(model, data, config, device):
    """
    Calcula a loss total combinando PDE, IC e BC.
    """
    pde_input, ic_input, ic_targets, bc_inputs, bc_targets = data
    
    # 1. Loss da PDE (Resíduo)
    residual = compute_pde_residual(model, pde_input, config)
    loss_pde = torch.mean(residual**2)
    
    # 2. Loss das Condições Iniciais (IC)
    u_pred_ic, v_pred_ic = compute_ic_derivatives(model, ic_input)
    
    # Loss para u(x, 0)
    loss_ic_u = torch.mean((u_pred_ic - ic_targets['u'])**2)
    # Loss para u_t(x, 0)
    loss_ic_v = torch.mean((v_pred_ic - ic_targets['v'])**2)
    
    loss_ic = loss_ic_u + loss_ic_v
    
    # 3. Loss das Condições de Contorno (BC)
    u_pred_bc_left = model(bc_inputs['left'])
    u_pred_bc_right = model(bc_inputs['right'])
    
    loss_bc = torch.mean((u_pred_bc_left - bc_targets['left'])**2) + \
              torch.mean((u_pred_bc_right - bc_targets['right'])**2)
              
    # Loss Total Ponderada
    total_loss = (config.W_PDE * loss_pde +
                  config.W_IC_U * loss_ic_u +
                  config.W_IC_V * loss_ic_v +
                  config.W_BC * loss_bc)
    
    return total_loss, loss_pde.item(), loss_ic.item(), loss_bc.item()


def run_training(config):
    """
    Executa o loop de treinamento principal.
    """
    set_seed(42)
    device = setup_device(config)
    
    model = PINN(config.LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1000, min_lr=1e-6)

    history = []
    best_loss = float('inf')

    print(f"Iniciando treinamento para o modelo: {config.MODEL_TYPE}")
    print(f"Dispositivo: {device}")
    
    pbar = tqdm(range(config.EPOCHS), desc="Treinando")
    for epoch in pbar:
        model.train()
        
        # Amostra novos pontos a cada época
        data = get_training_data(config, device)
        
        optimizer.zero_grad()
        total_loss, loss_pde, loss_ic, loss_bc = compute_loss(model, data, config, device)
        total_loss.backward()
        optimizer.step()
        
        scheduler.step(total_loss)
        
        # Salva o histórico
        history.append([epoch, total_loss.item(), loss_pde, loss_ic, loss_bc])

        # Atualiza a barra de progresso
        if epoch % 100 == 0:
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.2e}',
                'PDE': f'{loss_pde:.2e}',
                'IC': f'{loss_ic:.2e}',
                'BC': f'{loss_bc:.2e}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })

        # Salva o melhor modelo
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            save_model(model, config)
            
    print(f"Treinamento concluído. Melhor loss: {best_loss:.4e}")
    
    # Salva o histórico de treinamento
    history_df = pd.DataFrame(history, columns=['Epoch', 'Total Loss', 'PDE Loss', 'IC Loss', 'BC Loss'])
    save_training_history(history_df, config)
    
    return model, history_df