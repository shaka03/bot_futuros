import numpy as np
import torch

class ReplayBuffer:
    """
    Simple buffer para almacenar y muestrear transiciones para múltiples agentes.
    Cada transición incluye estado, acción, recompensa, siguiente estado y done.
    """
    def __init__(self, capacity, n_agents, state_dim, action_dim, device):
        """
        Inicializa el buffer de repetición.
        
        Args:
            capacity (int): Capacidad máxima del buffer.
            n_agents (int): Número de agentes.
            state_dim (int): Dimensión del estado.
            action_dim (int): Dimensión de la acción.
            device (torch.device): Dispositivo para los tensores.
        """
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.n_agents = n_agents
        
        self.states = np.zeros((capacity, n_agents, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_agents, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """
        Añade una transición al buffer.
        
        Args:
            state (np.array): Estado actual.
            action (np.array): Acción tomada.
            reward (np.array): Recompensa recibida.
            next_state (np.array): Siguiente estado.
            done (float): Indicador de si el episodio ha terminado.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Muestra un batch de transiciones del buffer.

        Args:
            batch_size (int): Tamaño del batch a muestrear.

        Returns:
            tuple: Tensores de estados, acciones, recompensas, siguientes estados y dones.
        """
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (torch.FloatTensor(self.states[idx]).to(self.device),
                torch.FloatTensor(self.actions[idx]).to(self.device),
                torch.FloatTensor(self.rewards[idx]).to(self.device),
                torch.FloatTensor(self.next_states[idx]).to(self.device),
                torch.FloatTensor(self.dones[idx]).to(self.device))