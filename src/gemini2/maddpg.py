import torch
import torch.nn.functional as F
import numpy as np
from config import Config
from networks import Actor, Critic
from buffer import ReplayBuffer

class Agent:
    """
    Clase que representa un agente MADDPG con su actor y crítico.
    """
    def __init__(self, state_dim, action_dim, all_state_dim, all_action_dim):
        """
        Inicializa el agente con sus redes actor y crítico.
        
        Args:
            state_dim (int): Dimensión del estado del agente.
            action_dim (int): Dimensión de la acción del agente.
            all_state_dim (int): Dimensión del estado global (todos los agentes).
            all_action_dim (int): Dimensión de la acción global (todos los agentes).
        """
        self.actor = Actor(state_dim, action_dim, hidden_dim=Config.HIDDEN_DIM).to(Config.DEVICE)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim=Config.HIDDEN_DIM).to(Config.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=Config.LR_ACTOR)

        self.critic = Critic(all_state_dim, all_action_dim, hidden_dim=Config.HIDDEN_DIM).to(Config.DEVICE)
        self.critic_target = Critic(all_state_dim, all_action_dim, hidden_dim=Config.HIDDEN_DIM).to(Config.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=Config.LR_CRITIC)

    def select_action(self, state, noise=0.0):
        """
        Selecciona una acción basada en el estado actual del agente.
        
        Args:
            state (np.array): Estado actual del agente.
            noise (float): Magnitud del ruido a añadir para exploración.
        
        Returns:
            np.array: Acción seleccionada.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
        action = self.actor(state).cpu().data.numpy()[0]
        if noise != 0:
            action += noise * np.random.randn(*action.shape)
        return np.clip(action, -1, 1)

class MADDPG:
    """
    Clase que representa el algoritmo MADDPG para múltiples agentes.
    """
    def __init__(self, n_agents, state_dims, action_dims):
        """
        Inicializa el MADDPG con múltiples agentes.
        
        Args:
            n_agents (int): Número de agentes.
            state_dims (list): Lista de dimensiones de estado para cada agente.
            action_dims (list): Lista de dimensiones de acción para cada agente.
        """
        self.agents = []
        self.n_agents = n_agents
        
        all_state_dim = sum(state_dims)
        all_action_dim = sum(action_dims)
        
        for i in range(n_agents):
            self.agents.append(Agent(state_dims[i], action_dims[i], all_state_dim, all_action_dim))
        
        self.memory = ReplayBuffer(Config.BUFFER_CAPACITY, n_agents, state_dims[0], action_dims[0], Config.DEVICE)

    def update(self, batch_size):
        """
        Actualiza las redes actor y crítico de todos los agentes.
        
        Args:
            batch_size (int): Tamaño del lote para el entrenamiento.
        """
        if self.memory.size < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Aplanar tensores para el Crítico (Global State/Action)
        state_batch_flat = states.view(batch_size, -1)
        action_batch_flat = actions.view(batch_size, -1)
        next_state_batch_flat = next_states.view(batch_size, -1)
        
        for agent_idx, agent in enumerate(self.agents):
            # 1. Update Critic
            with torch.no_grad():
                next_actions_list = [ag.actor_target(next_states[:, i, :]) for i, ag in enumerate(self.agents)]
                next_actions_flat = torch.cat(next_actions_list, dim=1)
                
                target_Q1, target_Q2 = agent.critic_target(next_state_batch_flat, next_actions_flat)
                target_Q = torch.min(target_Q1, target_Q2)
                y = rewards[:, agent_idx].unsqueeze(1) + Config.GAMMA * target_Q * (1 - dones)
            
            current_Q1, current_Q2 = agent.critic(state_batch_flat, action_batch_flat)
            critic_loss = F.mse_loss(current_Q1, y) + F.mse_loss(current_Q2, y)
            
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # 2. Update Actor
            curr_actions_list = []
            for i, ag in enumerate(self.agents):
                s_agent = states[:, i, :]
                if i == agent_idx:
                    curr_actions_list.append(ag.actor(s_agent))
                else:
                    curr_actions_list.append(ag.actor(s_agent).detach())
            
            curr_actions_flat = torch.cat(curr_actions_list, dim=1)
            actor_loss = -agent.critic.Q1(state_batch_flat, curr_actions_flat).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # 3. Soft Update
            for param, target_param in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                target_param.data.copy_(Config.TAU * param.data + (1 - Config.TAU) * target_param.data)
            for param, target_param in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                target_param.data.copy_(Config.TAU * param.data + (1 - Config.TAU) * target_param.data)