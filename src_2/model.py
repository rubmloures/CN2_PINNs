# src_2/model.py
import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Rede Neural simples (MLP) para a PINN.
    COM normalização de entrada.
    """
    def __init__(self, layers, x_bounds, y_bounds, t_bounds):
        """
        Inicializa a rede neural.
        :param layers: Lista de neurônios por camada.
        :param x_bounds: Lista [min, max] para x.
        :param y_bounds: Lista [min, max] para y.
        :param t_bounds: Lista [min, max] para t.
        """
        super(PINN, self).__init__()
        
        # Armazena os limites para normalização
        self.register_buffer('x_min', torch.tensor(x_bounds[0]))
        self.register_buffer('x_max', torch.tensor(x_bounds[1]))
        self.register_buffer('y_min', torch.tensor(y_bounds[0]))
        self.register_buffer('y_max', torch.tensor(y_bounds[1]))
        self.register_buffer('t_min', torch.tensor(t_bounds[0]))
        self.register_buffer('t_max', torch.tensor(t_bounds[1]))

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = nn.Tanh()
        self.init_weights()

    def normalize(self, x_in, min_val, max_val):
        """Normaliza a entrada para o intervalo [-1, 1]"""
        return 2.0 * (x_in - min_val) / (max_val - min_val) - 1.0

    def forward(self, x):
        """
        Forward pass.
        :param x: Tensor de entrada (ex: [x, y, t])
        :return: Tensor de saída (ex: u(x, y, t))
        """
        # Extrai e normaliza as entradas
        x_norm = self.normalize(x[:, 0:1], self.x_min, self.x_max)
        y_norm = self.normalize(x[:, 1:2], self.y_min, self.y_max)
        t_norm = self.normalize(x[:, 2:3], self.t_min, self.t_max)
        
        # Concatena as entradas normalizadas
        x_normalized = torch.cat((x_norm, y_norm, t_norm), dim=1)

        # Passa pela rede
        for i, layer in enumerate(self.layers):
            x_normalized = layer(x_normalized)
            if i < len(self.layers) - 1:
                x_normalized = self.activation(x_normalized)
        
        return x_normalized

    def init_weights(self):
        """
        Inicialização dos pesos usando Xavier.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)