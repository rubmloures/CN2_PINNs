# src/data_loader.py
import torch

def get_training_data(config, device):
    """
    Gera os pontos de treinamento (colocação, inicial, contorno)
    para a equação da onda 1D.
    """
    x_min, x_max = config.X_BOUNDS
    t_min, t_max = config.T_BOUNDS

    # 1. Pontos de Condição Inicial (IC) - (t = t_min)
    x_ic = torch.rand((config.N_IC, 1), device=device) * (x_max - x_min) + x_min
    t_ic = torch.full_like(x_ic, t_min)
    ic_input = torch.cat((x_ic, t_ic), dim=1)
    
    # Condição inicial: u(x, 0) = pulso Gaussiano
    # u(x, 0) = exp(-a * (x - centro)^2)
    center = (x_max + x_min) / 2
    a = 100.0
    u_target_ic = torch.exp(-a * (x_ic - center)**2)
    
    # Condição inicial de velocidade: u_t(x, 0) = 0 (começa em repouso)
    v_target_ic = torch.zeros_like(u_target_ic)
    
    ic_targets = {'u': u_target_ic, 'v': v_target_ic}

    # 2. Pontos de Condição de Contorno (BC) - (x = x_min, x = x_max)
    # Condições de Dirichlet (pontas fixas): u(0, t) = 0, u(L, t) = 0
    t_bc = torch.rand((config.N_BC, 1), device=device) * (t_max - t_min) + t_min
    
    # Contorno esquerdo (x = x_min)
    x_bc_left = torch.full_like(t_bc, x_min)
    bc_input_left = torch.cat((x_bc_left, t_bc), dim=1)
    
    # Contorno direito (x = x_max)
    x_bc_right = torch.full_like(t_bc, x_max)
    bc_input_right = torch.cat((x_bc_right, t_bc), dim=1)

    bc_inputs = {'left': bc_input_left, 'right': bc_input_right}
    
    # Alvo para ambos os contornos é 0
    u_target_bc = torch.zeros_like(t_bc)
    bc_targets = {'left': u_target_bc, 'right': u_target_bc}

    # 3. Pontos de Colocação (PDE) - (x, t) dentro do domínio
    # Amostragem aleatória (pode ser trocada por Latin Hypercube)
    x_pde = torch.rand((config.N_PDE, 1), device=device) * (x_max - x_min) + x_min
    t_pde = torch.rand((config.N_PDE, 1), device=device) * (t_max - t_min) + t_min
    
    # Habilita o cálculo de gradientes para esses tensores
    pde_input = torch.cat((x_pde, t_pde), dim=1).requires_grad_(True)
    
    return pde_input, ic_input, ic_targets, bc_inputs, bc_targets