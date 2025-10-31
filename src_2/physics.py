# src_2/physics.py
import torch

def get_velocity(x, y, config):
    """
    Retorna o valor da velocidade 'c' com base na configuração.
    Para 2D, c(x, y).
    """
    if config.MODEL_TYPE == "simulacao_2d":
        # c(x, y) = C_BASE + C_GRAD_X * x + C_GRAD_Y * y
        return config.C_BASE + config.C_GRAD_X * x + config.C_GRAD_Y * y
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {config.MODEL_TYPE}")

def compute_pde_residual(model, pde_input, config):
    """
    Calcula o resíduo da Equação da Onda 2D:
    Resíduo = u_tt - c^2 * (u_xx + u_yy)
    """
    # pde_input é [x, y, t]
    x = pde_input[:, 0:1]
    y = pde_input[:, 1:2]
    # t = pde_input[:, 2:3] # Não é necessário para o resíduo em si, mas sim para as derivadas

    u = model(pde_input)
    
    # Derivadas de primeira ordem
    u_grads = torch.autograd.grad(u, pde_input, 
                                 grad_outputs=torch.ones_like(u), 
                                 create_graph=True)[0]
    u_x = u_grads[:, 0:1]
    u_y = u_grads[:, 1:2]
    u_t = u_grads[:, 2:3]

    # Derivadas de segunda ordem
    # u_xx
    u_xx_grads = torch.autograd.grad(u_x, pde_input, 
                                     grad_outputs=torch.ones_like(u_x), 
                                     create_graph=True)[0]
    u_xx = u_xx_grads[:, 0:1] # Componente x

    # u_yy
    u_yy_grads = torch.autograd.grad(u_y, pde_input, 
                                     grad_outputs=torch.ones_like(u_y), 
                                     create_graph=True)[0]
    u_yy = u_yy_grads[:, 1:2] # Componente y

    # u_tt
    u_tt_grads = torch.autograd.grad(u_t, pde_input, 
                                     grad_outputs=torch.ones_like(u_t), 
                                     create_graph=True)[0]
    u_tt = u_tt_grads[:, 2:3] # Componente t
    
    # Obter velocidade c(x, y)
    c = get_velocity(x, y, config)
    
    # Calcular o resíduo da PDE
    residual = u_tt - (c**2) * (u_xx + u_yy)
    
    return residual

def compute_ic_derivatives(model, ic_input):
    """
    Calcula u(x,y,0) e a derivada temporal u_t(x,y,0) 
    para a loss da condição inicial.
    """
    # ic_input é [x, y, t=0]
    ic_input.requires_grad_(True)
    
    u_pred = model(ic_input)
    
    # Precisamos de u_t(x, y, 0)
    u_t_pred_grads = torch.autograd.grad(u_pred, ic_input, 
                                          grad_outputs=torch.ones_like(u_pred), 
                                          create_graph=True)[0]
    # Pega a derivada em t, que agora é o índice 2
    u_t_pred = u_t_pred_grads[:, 2:3] 

    return u_pred, u_t_pred