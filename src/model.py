# src/model.py
import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Rede Neural simples (MLP) para a PINN.
    """
    def __init__(self, layers):
        """
        Inicializa a rede neural.
        :param layers: Lista contendo o número de neurônios em cada camada.
                       Ex: [2, 32, 32, 1] para 2 entradas, 2 camadas ocultas com 32 neurônios, 1 saída.
        """
        super(PINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Usamos Tanh como função de ativação, é comum em PINNs
        # por ser infinitamente diferenciável.
        self.activation = nn.Tanh()

        self.init_weights()

    def forward(self, x):
        """
        Forward pass.
        :param x: Tensor de entrada (ex: [x, t])
        :return: Tensor de saída (ex: u(x, t))
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def init_weights(self):
        """
        Inicialização dos pesos usando Xavier.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

# Modelo para equação da onda 2D
class PINN2D(nn.Module):
    """PINN para equação da onda 2D"""
    
    def __init__(self, layers=[3, 50, 50, 50, 50, 1], activation='tanh'):
        super(PINN2D, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Função de ativação
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.tanh
    
    def forward(self, x, y, t):
        # Concatenar entradas: [x, y, t]
        inputs = torch.cat([x, y, t], dim=1)
        
        # Forward pass através das camadas
        for i, layer in enumerate(self.layers[:-1]):
            inputs = self.activation(layer(inputs))
        
        # Camada final (linear)
        output = self.layers[-1](inputs)
        
        return output