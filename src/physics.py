# src/physics.py
import torch

def get_velocity(x, config):
    """
    Retorna o valor da velocidade 'c' com base na configuração.
    Para 'constante', é um escalar.
    Para 'variavel', é c(x).
    """
    if config.MODEL_TYPE == "constante":
        return config.C
    elif config.MODEL_TYPE == "variavel":
        # c(x) = C_BASE + C_GRAD * x
        return config.C_BASE + config.C_GRAD * x
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {config.MODEL_TYPE}")

def compute_pde_residual(model, pde_input, config):
    """
    Calcula o resíduo da Equação da Onda 1D:
    Resíduo = u_tt - c^2 * u_xx
    """
    # pde_input é o tensor [x, t]
    
    # Precisamos de 'x' separado para o caso de c(x)
    x = pde_input[:, 0:1]
    
    u = model(pde_input)
    
    # Calcular derivadas usando torch.autograd.grad
    
    # u_t e u_x (Primeiras derivadas)
    # Derivamos u em relação a pde_input
    u_grads = torch.autograd.grad(u, pde_input, 
                                 grad_outputs=torch.ones_like(u), 
                                 create_graph=True)[0]
    u_x = u_grads[:, 0:1]
    u_t = u_grads[:, 1:2]

    # u_tt (Segunda derivada temporal)
    # Derivamos u_t (que é du/dt) em relação a pde_input
    u_tt_grads = torch.autograd.grad(u_t, pde_input, 
                                      grad_outputs=torch.ones_like(u_t), 
                                      create_graph=True)[0]
    # E pegamos a componente temporal (índice 1) do resultado
    u_tt = u_tt_grads[:, 1:2]

    # u_xx (Segunda derivada espacial)
    # Derivamos u_x (que é du/dx) em relação a pde_input
    u_xx_grads = torch.autograd.grad(u_x, pde_input, 
                                      grad_outputs=torch.ones_like(u_x), 
                                      create_graph=True)[0]
    # E pegamos a componente espacial (índice 0) do resultado
    u_xx = u_xx_grads[:, 0:1]
    
    # Obter velocidade c (constante ou variável c(x))
    c = get_velocity(x, config)
    
    # Calcular o resíduo da PDE
    residual = u_tt - (c**2) * u_xx
    
    return residual

def compute_ic_derivatives(model, ic_input):
    """
    Calcula u(x,0) e a derivada temporal u_t(x,0) 
    para a loss da condição inicial.
    """
    # Precisamos explicitamente habilitar o gradiente em ic_input
    # para calcular a derivada u_t
    ic_input.requires_grad_(True)
    
    u_pred = model(ic_input)
    
    # Precisamos de u_t(x, 0)
    u_t_pred_grads = torch.autograd.grad(u_pred, ic_input, 
                                          grad_outputs=torch.ones_like(u_pred), 
                                          create_graph=True)[0]
    u_t_pred = u_t_pred_grads[:, 1:2] # Pega só a derivada em t

    return u_pred, u_t_pred