import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, n_agents, state_dim, action_dim, device):
        """
        Buffer de repetición priorizado para MADDPG.
        Args:
            capacity: Tamaño máximo del buffer.
            n_agents: Número de agentes (contratos).
            state_dim: Dimensión del estado (5 en este caso).
            action_dim: Dimensión de la acción (2 en este caso).
            device: Dispositivo torch (cuda/cpu).
        """
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Inicialización de buffers con memoria pre-reservada
        # Forma: (Capacidad, Agentes, Dimensiones)
        self.states = np.zeros((capacity, n_agents, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_agents, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Agrega una transición al buffer."""
        # state: (n_agents, state_dim)
        # action: (n_agents, action_dim)
        # reward: (n_agents,) o lista
        
        self.states[self.ptr] = np.array(state)
        self.actions[self.ptr] = np.array(action)
        self.rewards[self.ptr] = np.array(reward)
        self.next_states[self.ptr] = np.array(next_state)
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Muestrea un lote aleatorio de transiciones."""
        idx = np.random.choice(self.size, batch_size, replace=False)
        
        # Convertir directamente a Tensores y mover al dispositivo
        states = torch.FloatTensor(self.states[idx]).to(self.device)
        actions = torch.FloatTensor(self.actions[idx]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idx]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idx]).to(self.device)
        dones = torch.FloatTensor(self.dones[idx]).to(self.device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size