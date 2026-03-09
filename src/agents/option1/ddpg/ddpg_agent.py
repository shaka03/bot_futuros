"""DDPG con redes LSTM para estados secuenciales.

Incluye:
- ActorLSTM
- CriticLSTM
- SequenceReplayBuffer
- DDPGAgent
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import CONFIG, ProjectConfig


# ---------------------------------------------------------------------
# 1) Actor
# ---------------------------------------------------------------------
class ActorLSTM(nn.Module):
    """Actor basado en LSTM.

    Input:
        state shape = (batch_size, sequence_length, num_features)
    Output:
        action shape = (batch_size, action_dim), rango [-1, 1]
    """

    def __init__(
        self,
        num_features: int,
        action_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward del actor."""
        lstm_out, _ = self.lstm(state)         # (B, T, H)
        last_timestep = lstm_out[:, -1, :]     # (B, H)

        x = torch.relu(self.fc1(last_timestep))
        action = self.tanh(self.fc2(x))        # [-1, 1]
        return action


# ---------------------------------------------------------------------
# 2) Critic
# ---------------------------------------------------------------------
class CriticLSTM(nn.Module):
    """Crítico basado en LSTM + MLP."""

    def __init__(
        self,
        num_features: int,
        action_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward del crítico."""
        lstm_out, _ = self.lstm(state)         # (B, T, H)
        last_timestep = lstm_out[:, -1, :]     # (B, H)

        x = torch.cat([last_timestep, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# ---------------------------------------------------------------------
# 3) Replay Buffer secuencial
# ---------------------------------------------------------------------
class SequenceReplayBuffer:
    """Replay Buffer para estados secuenciales (T, F), optimizado en NumPy."""

    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        num_features: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.num_features = int(num_features)
        self.action_dim = int(action_dim)
        self.device = device

        self.states = np.zeros((self.capacity, self.sequence_length, self.num_features), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.sequence_length, self.num_features), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr: int = 0
        self.size: int = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Agrega transición al buffer."""
        idx = self.ptr

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx, 0] = float(reward)
        self.next_states[idx] = next_state
        self.dones[idx, 0] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Muestrea batch y devuelve tensores listos para device."""
        if self.size < batch_size:
            raise ValueError(f"No hay suficientes muestras: size={self.size}, batch={batch_size}")

        idxs = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[idxs]).to(self.device)
        actions = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idxs]).to(self.device)
        dones = torch.from_numpy(self.dones[idxs]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------
# 4) Agente DDPG
# ---------------------------------------------------------------------
class DDPGAgent:
    """Agente DDPG con redes target y soft update."""

    def __init__(
        self,
        num_features: int,
        action_dim: int = 6,
        config: ProjectConfig = CONFIG,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.action_dim = int(action_dim)

        # Device
        if device is not None:
            self.device = torch.device(device)
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        hidden_size = self.config.lstm.hidden_size
        num_layers = self.config.lstm.num_layers
        dropout = self.config.lstm.dropout

        # Redes main
        self.actor = ActorLSTM(num_features, self.action_dim, hidden_size, num_layers, dropout=dropout).to(self.device)
        self.critic = CriticLSTM(num_features, self.action_dim, hidden_size, num_layers, dropout=dropout).to(self.device)

        # Redes target
        self.actor_target = ActorLSTM(num_features, self.action_dim, hidden_size, num_layers, dropout=dropout).to(self.device)
        self.critic_target = CriticLSTM(num_features, self.action_dim, hidden_size, num_layers, dropout=dropout).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_target.eval()
        self.critic_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.ddpg.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.ddpg.critic_lr)

        self.gamma = float(self.config.ddpg.gamma)
        self.tau = float(self.config.ddpg.tau)
        self.batch_size = int(self.config.ddpg.batch_size)

        # Ruido (alineado con config.py limpio)
        self.noise_std_init = float(self.config.ddpg.exploration_noise_std)
        self.noise_std = float(self.config.ddpg.exploration_noise_std)
        self.noise_std_min = float(self.config.ddpg.exploration_noise_min_std)
        self.noise_decay = float(self.config.ddpg.exploration_noise_decay)

        self.replay_buffer = SequenceReplayBuffer(
            capacity=self.config.ddpg.buffer_capacity,
            sequence_length=self.config.lstm.sequence_length,
            num_features=num_features,
            action_dim=self.action_dim,
            device=self.device,
        )

        self.mse_loss = nn.MSELoss()

    def reset_noise(self) -> None:
        self.noise_std = self.noise_std_init

    def decay_noise(self) -> None:
        self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay)

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Selecciona acción usando actor principal."""
        self.actor.eval()

        state_np = state.astype(np.float32)
        if state_np.ndim == 2:
            state_np = np.expand_dims(state_np, axis=0)  # (T, F) -> (1, T, F)

        state_t = torch.from_numpy(state_np).to(self.device)
        action_t = self.actor(state_t)
        action = action_t.squeeze(0).cpu().numpy()

        if add_noise:
            noise = np.random.normal(0.0, self.noise_std, size=self.action_dim).astype(np.float32)
            action = action + noise

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def soft_update(self) -> None:
        """Soft update de redes target."""
        for target_p, main_p in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1.0 - self.tau) * target_p.data)

        for target_p, main_p in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1.0 - self.tau) * target_p.data)

    def train_step(self) -> Dict[str, float]:
        """Ejecuta un paso de entrenamiento DDPG."""
        if len(self.replay_buffer) < self.batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target_next = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * (1.0 - dones) * q_target_next

        q_pred = self.critic(states, actions)
        critic_loss = self.mse_loss(q_pred, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Guarda transición en replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)