# src_2/data_loader.py
import torch
import numpy as np

def get_training_data(config, device):
    """
    Gera os pontos de treinamento (colocação, inicial, contorno)
    para a equação da onda 2D.
    """
    x_min, x_max = config.X_BOUNDS
    y_min, y_max = config.Y_BOUNDS
    t_min, t_max = config.T_BOUNDS

    # 1. Pontos de Condição Inicial (IC) - (t = t_min)
    x_ic = torch.rand((config.N_IC, 1), device=device) * (x_max - x_min) + x_min
    y_ic = torch.rand((config.N_IC, 1), device=device) * (y_max - y_min) + y_min
    t_ic = torch.full_like(x_ic, t_min)
    ic_input = torch.cat((x_ic, y_ic, t_ic), dim=1)
    
    # Condição inicial: u(x, y, 0) = pulso Gaussiano 2D
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    a = 50.0 # Largura do pulso
    u_target_ic = torch.exp(-a * ((x_ic - center_x)**2 + (y_ic - center_y)**2))
    
    # Condição inicial de velocidade: u_t(x, y, 0) = 0
    v_target_ic = torch.zeros_like(u_target_ic)
    
    ic_targets = {'u': u_target_ic, 'v': v_target_ic}

    # 2. Pontos de Condição de Contorno (BC) - 4 bordas
    # Condições de Dirichlet (bordas fixas): u=0 em todas as bordas
    
    n_bc_edge = config.N_BC # Pontos por borda
    t_bc = torch.rand((n_bc_edge, 1), device=device) * (t_max - t_min) + t_min
    
    # Bordas x=x_min (esquerda) e x=x_max (direita)
    y_bc_sides = torch.rand((n_bc_edge, 1), device=device) * (y_max - y_min) + y_min
    x_bc_left = torch.full_like(t_bc, x_min)
    x_bc_right = torch.full_like(t_bc, x_max)
    bc_input_left = torch.cat((x_bc_left, y_bc_sides, t_bc), dim=1)
    bc_input_right = torch.cat((x_bc_right, y_bc_sides, t_bc), dim=1)

    # Bordas y=y_min (baixo) e y=y_max (cima)
    x_bc_topbot = torch.rand((n_bc_edge, 1), device=device) * (x_max - x_min) + x_min
    y_bc_bottom = torch.full_like(t_bc, y_min)
    y_bc_top = torch.full_like(t_bc, y_max)
    bc_input_bottom = torch.cat((x_bc_topbot, y_bc_bottom, t_bc), dim=1)
    bc_input_top = torch.cat((x_bc_topbot, y_bc_top, t_bc), dim=1)

    bc_inputs = {'left': bc_input_left, 'right': bc_input_right, 
                 'bottom': bc_input_bottom, 'top': bc_input_top}
    
    # Alvo para todas as bordas é 0
    u_target_bc = torch.zeros((n_bc_edge, 1), device=device)
    bc_targets = {'left': u_target_bc, 'right': u_target_bc, 
                  'bottom': u_target_bc, 'top': u_target_bc}

    # 3. Pontos de Colocação (PDE) - (x, y, t) dentro do domínio
    x_pde = torch.rand((config.N_PDE, 1), device=device) * (x_max - x_min) + x_min
    y_pde = torch.rand((config.N_PDE, 1), device=device) * (y_max - y_min) + y_min
    t_pde = torch.rand((config.N_PDE, 1), device=device) * (t_max - t_min) + t_min
    
    pde_input = torch.cat((x_pde, y_pde, t_pde), dim=1).requires_grad_(True)
    
    return pde_input, ic_input, ic_targets, bc_inputs, bc_targets