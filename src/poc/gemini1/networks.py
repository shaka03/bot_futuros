import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Red neuronal para el Actor en DDPG/SAC.
    Toma el estado como entrada y devuelve la acción.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize the Actor network.
        
        Args:
            state_dim (int): Dimensión del espacio de estados.
            action_dim (int): Dimensión del espacio de acciones.
            hidden_dim (int): Número de neuronas en las capas ocultas.
        """
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Propaga el estado a través de la red para obtener la acción.
        
        Args:
            state (torch.Tensor): Tensor que representa el estado actual.
        
        Returns:
            torch.Tensor: Tensor que representa la acción tomada.
        """
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # Tanh para acotar la salida entre -1 y 1 (Hedge Ratio y Decisión)
        return torch.tanh(self.l3(x))

class Critic(nn.Module):
    """
    Red neuronal para el Crítico en DDPG/SAC.
    Toma el estado y la acción como entrada y devuelve el valor Q.
    """
    def __init__(self, all_states_dim, all_actions_dim, hidden_dim=256):
        """
        Inicializa la red del Crítico.
        Args:
            all_states_dim (int): Dimensión del espacio de estados combinado.
            all_actions_dim (int): Dimensión del espacio de acciones combinado.
            hidden_dim (int): Número de neuronas en las capas ocultas.
        """
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(all_states_dim + all_actions_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture (Dual Q)
        self.l4 = nn.Linear(all_states_dim + all_actions_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        """
        Propaga el estado y la acción a través de la red para obtener los valores Q.
        
        Args:
            state (torch.Tensor): Tensor que representa el estado actual.
            action (torch.Tensor): Tensor que representa la acción tomada.
        
        Returns:
            torch.Tensor, torch.Tensor: Valores Q1 y Q2.
        """
        xu = torch.cat([state, action], 1)
        
        # Q1
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        
        # Q2
        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        
        return x1, x2

    def Q1(self, state, action):
        """
        Obtiene solo el valor Q1 para un estado y acción dados.
        
        Args:
            state (torch.Tensor): Tensor que representa el estado actual.
            action (torch.Tensor): Tensor que representa la acción tomada.
        
        Returns:
            torch.Tensor: Valor Q1.
        """
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1