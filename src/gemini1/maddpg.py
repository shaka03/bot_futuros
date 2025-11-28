import torch
import torch.nn.functional as F
import numpy as np
from config import Config
from networks import Actor, Critic
from buffer import ReplayBuffer

class Agent:
    """
    Clase que representa un agente MADDPG con su Actor y Crítico.
    """
    def __init__(self, state_dim, action_dim, all_state_dim, all_action_dim):
        """
        Inicializa el agente con sus redes y optimizadores.
        
        Args:
            state_dim (int): Dimensión del espacio de estados del agente.
            action_dim (int): Dimensión del espacio de acciones del agente.
            all_state_dim (int): Dimensión del espacio de estados global (todos los agentes).
            all_action_dim (int): Dimensión del espacio de acciones global (todos los agentes).
        """
        self.actor = Actor(state_dim, action_dim).to(Config.DEVICE)
        self.actor_target = Actor(state_dim, action_dim).to(Config.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=Config.LR_ACTOR)

        self.critic = Critic(all_state_dim, all_action_dim).to(Config.DEVICE)
        self.critic_target = Critic(all_state_dim, all_action_dim).to(Config.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=Config.LR_CRITIC)

    def select_action(self, state, noise=0.0):
        """
        Selecciona una acción para el estado dado, añadiendo ruido si es necesario.
        
        Args:
            state (np.array): Estado actual del agente.
            noise (float): Magnitud del ruido a añadir a la acción.
        
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
    Clase que representa el sistema MADDPG con múltiples agentes.
    """
    def __init__(self, n_agents, state_dims, action_dims):
        """
        Inicializa el sistema MADDPG con múltiples agentes y un buffer de replay.
        
        Args:
            n_agents (int): Número de agentes.
            state_dims (list): Lista con las dimensiones de los estados para cada agente.
            action_dims (list): Lista con las dimensiones de las acciones para cada agente.
        """
        self.agents = []
        self.n_agents = n_agents

        state_dim_single = state_dims[0]
        action_dim_single = action_dims[0]
        
        all_state_dim = sum(state_dims)
        all_action_dim = sum(action_dims)
        
        for i in range(n_agents):
            self.agents.append(Agent(state_dims[i], action_dims[i], all_state_dim, all_action_dim))
            
        # Inicializar ReplayBuffer
        self.memory = ReplayBuffer(
            capacity=Config.BUFFER_SIZE, 
            n_agents=n_agents, 
            state_dim=state_dim_single, 
            action_dim=action_dim_single,
            device=Config.DEVICE
        )

        self.batch_size = Config.BATCH_SIZE
        self.gamma = Config.GAMMA
        self.tau = Config.TAU

    def update(self):
        """
        Actualiza las redes de los agentes usando muestras del buffer de replay.
        
        Steps:
            1. Muestra un batch del buffer de replay.
            2. Actualiza la red crítica de cada agente.
            3. Actualiza la red actor de cada agente.
            4. Realiza una actualización suave de las redes objetivo.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Aplanar para los Críticos (que ven todo el estado global)
        # Reshape de (Batch, Agents, Dim) -> (Batch, Agents*Dim)
        state_batch_flat = states.view(self.batch_size, -1)
        next_state_batch_flat = next_states.view(self.batch_size, -1)
        action_batch_flat = actions.view(self.batch_size, -1)
        
        for agent_idx, agent in enumerate(self.agents):
            # 1. Update Critic
            with torch.no_grad():
                next_actions = []
                for i, ag in enumerate(self.agents):
                    # Extraer estado del agente i del batch
                    next_s_agent = next_states[:, i, :]
                    next_actions.append(ag.actor_target(next_s_agent))
                
                next_actions = torch.cat(next_actions, dim=1)
                
                target_Q1, target_Q2 = agent.critic_target(next_state_batch_flat, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                
                # Seleccionar recompensa solo para este agente
                # rewards shape: (Batch, Agents) -> slice (Batch, 1)
                r = rewards[:, agent_idx].unsqueeze(1)
                
                # Dones es (Batch, 1), aplica a todos
                y = r + (1 - dones) * self.gamma * target_Q

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
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_memory(self, obs, act, rew, next_obs, done):
        """
        Añade una transición al buffer de replay.
        
        Args:
            obs (np.array): Observaciones actuales de todos los agentes.
            act (np.array): Acciones tomadas por todos los agentes.
            rew (np.array): Recompensas recibidas por todos los agentes.
            next_obs (np.array): Observaciones siguientes de todos los agentes.
            done (np.array): Indicadores de finalización para todos los agentes.
        """
        # Delegar al nuevo buffer
        self.memory.add(obs, act, rew, next_obs, done)