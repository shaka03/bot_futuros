import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Red neuronal para el actor en un entorno continuo.
    Toma el estado como entrada y devuelve la acción.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Inicializa la red neuronal del actor.
        
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
        Propagación hacia adelante de la red neuronal del actor.
        
        Args:
            state (torch.Tensor): Tensor que representa el estado actual.
            
        Returns:
            torch.Tensor: Tensor que representa la acción tomada.
        """
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x)) # Salida entre -1 y 1

class Critic(nn.Module):
    """
    Red neuronal para el crítico en un entorno continuo.
    Toma el estado y la acción como entrada y devuelve el valor Q.
    Implementa una arquitectura de doble crítico (Dual Q).
    """
    def __init__(self, all_states_dim, all_actions_dim, hidden_dim=256):
        """
        Inicializa la red neuronal del crítico.
        Args:
            all_states_dim (int): Dimensión combinada del espacio de estados de todos los agentes.
            all_actions_dim (int): Dimensión combinada del espacio de acciones de todos los agentes.
            hidden_dim (int): Número de neuronas en las capas ocultas.
        """
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(all_states_dim + all_actions_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 (Dual Q)
        self.l4 = nn.Linear(all_states_dim + all_actions_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        """
        Propagación hacia adelante de la red neuronal del crítico.
        
        Args:
            state (torch.Tensor): Tensor que representa el estado actual combinado de todos los agentes.
            action (torch.Tensor): Tensor que representa la acción tomada combinada de todos los agentes.
        
        Returns:
            torch.Tensor, torch.Tensor: Valores Q1 y Q2 correspondientes al estado y acción dados.
        """
        xu = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(xu))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(xu))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    def Q1(self, state, action):
        """
        Calcula solo el valor Q1 para el estado y acción dados.
        
        Args:
            state (torch.Tensor): Tensor que representa el estado actual combinado de todos los agentes.
            action (torch.Tensor): Tensor que representa la acción tomada combinada de todos los agentes.
        
        Returns:
            torch.Tensor: Valor Q1 correspondiente al estado y acción dados.
        """
        xu = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(xu))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)