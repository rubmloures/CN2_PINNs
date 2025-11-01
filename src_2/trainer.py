# src_2/trainer.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm

# Importa dos módulos locais (src_2)
from src_2.model import PINN
from src_2.data_loader import get_training_data
from src_2.physics import compute_pde_residual, compute_ic_derivatives
from src_2.utils import set_seed, setup_device, save_model, save_training_history

def compute_loss(model, data, config, device):
    """
    Calcula a loss total combinando PDE, IC e BC (4 bordas).
    (Nenhuma mudança nesta função)
    """
    pde_input, ic_input, ic_targets, bc_inputs, bc_targets = data
    
    # 1. Loss da PDE (Resíduo)
    residual = compute_pde_residual(model, pde_input, config)
    loss_pde = torch.mean(residual**2)
    
    # 2. Loss das Condições Iniciais (IC)
    u_pred_ic, v_pred_ic = compute_ic_derivatives(model, ic_input)
    
    loss_ic_u = torch.mean((u_pred_ic - ic_targets['u'])**2)
    loss_ic_v = torch.mean((v_pred_ic - ic_targets['v'])**2)
    loss_ic = loss_ic_u + loss_ic_v
    
    # 3. Loss das Condições de Contorno (BC) - 4 bordas
    u_pred_bc_left = model(bc_inputs['left'])
    u_pred_bc_right = model(bc_inputs['right'])
    u_pred_bc_bottom = model(bc_inputs['bottom'])
    u_pred_bc_top = model(bc_inputs['top'])
    
    loss_bc = (torch.mean((u_pred_bc_left - bc_targets['left'])**2) +
               torch.mean((u_pred_bc_right - bc_targets['right'])**2) +
               torch.mean((u_pred_bc_bottom - bc_targets['bottom'])**2) +
               torch.mean((u_pred_bc_top - bc_targets['top'])**2))
              
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
    
    # Passa os limites da configuração para o construtor do modelo
    model = PINN(config.LAYERS, 
                 config.X_BOUNDS, 
                 config.Y_BOUNDS, 
                 config.T_BOUNDS).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1000, min_lr=1e-6)

    history = []
    best_loss = float('inf')

    print(f"Iniciando treinamento para o modelo: {config.MODEL_TYPE}")
    print(f"Dispositivo: {device}")
    
    ckpt_every = getattr(config, 'CKPT_EVERY', 500)
    pbar = tqdm(range(config.EPOCHS), desc="Treinando")
    for epoch in pbar:
        model.train()
        # Gera novos dados de treino (sample collocation / IC / BC)
        data = get_training_data(config, device)

        optimizer.zero_grad()
        try:
            total_loss, loss_pde, loss_ic, loss_bc = compute_loss(model, data, config, device)
        except Exception as e:
            print(f"Erro ao calcular loss na época {epoch}: {e}")
            raise

        if not torch.isfinite(total_loss):
            print(f"Loss não finita detectada na época {epoch}: {total_loss}")
            break

        total_loss.backward()
        optimizer.step()

        scheduler.step(total_loss)

        history.append([epoch, total_loss.item(), loss_pde, loss_ic, loss_bc])

        # Atualiza barra e prints com menos frequência
        if epoch % getattr(config, 'LOG_EVERY', 50) == 0:
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.2e}',
                'PDE': f'{loss_pde:.2e}',
                'IC': f'{loss_ic:.2e}',
                'BC': f'{loss_bc:.2e}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })

        # Checkpoint intermediário
        if (epoch + 1) % ckpt_every == 0:
            try:
                save_model(model, config, suffix=f"epoch{epoch+1}")
                print(f"Checkpoint salvo na época {epoch+1}")
            except Exception as e:
                print(f"Falha ao salvar checkpoint na época {epoch+1}: {e}")

        # Atualiza melhor modelo
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            save_model(model, config)
            
    print(f"Treinamento concluído. Melhor loss: {best_loss:.4e}")
    
    history_df = pd.DataFrame(history, columns=['Epoch', 'Total Loss', 'PDE Loss', 'IC Loss', 'BC Loss'])
    save_training_history(history_df, config)
    
    return model, history_df