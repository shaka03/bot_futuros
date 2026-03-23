"""PPO con redes LSTM para estados secuenciales.

Incluye:
- ActorLSTM (política Gaussiana continua)
- CriticLSTM (valor V(s))
- RolloutBuffer on-policy
- PPOAgent (GAE + clipping objective)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal

from config import CONFIG, ProjectConfig


# ---------------------------------------------------------------------
# 1) Actor
# ---------------------------------------------------------------------
class ActorLSTM(nn.Module):
    """Actor LSTM para política Gaussiana continua.

    Input:
        state shape = (batch_size, sequence_length, num_features)
    Output:
        mu shape = (batch_size, action_dim), rango [-1, 1]
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
            dropout=dropout,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(state)            # (B, T, H)
        last_timestep = lstm_out[:, -1, :]        # (B, H)

        x = torch.relu(self.fc1(last_timestep))
        mu = self.tanh(self.fc_mu(x))             # [-1, 1]
        return mu


# ---------------------------------------------------------------------
# 2) Critic
# ---------------------------------------------------------------------
class CriticLSTM(nn.Module):
    """Crítico LSTM para estimar V(s)."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(state)            # (B, T, H)
        last_timestep = lstm_out[:, -1, :]        # (B, H)

        x = torch.relu(self.fc1(last_timestep))
        v = self.fc_v(x)                          # (B, 1)
        return v


# ---------------------------------------------------------------------
# 3) Rollout Buffer on-policy
# ---------------------------------------------------------------------
@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    """Buffer on-policy para PPO."""

    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        state: np.ndarray,    # (T, F)
        action: np.ndarray,   # (A,)
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.states.append(state.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def build_tensors(
        self,
        returns: np.ndarray,
        advantages: np.ndarray,
        device: torch.device,
    ) -> RolloutBatch:
        states_t = torch.from_numpy(np.asarray(self.states, dtype=np.float32)).to(device)        # (N,T,F)
        actions_t = torch.from_numpy(np.asarray(self.actions, dtype=np.float32)).to(device)      # (N,A)
        logprobs_t = torch.from_numpy(np.asarray(self.logprobs, dtype=np.float32)).to(device)    # (N,)
        returns_t = torch.from_numpy(returns.astype(np.float32)).to(device)                       # (N,)
        adv_t = torch.from_numpy(advantages.astype(np.float32)).to(device)                        # (N,)

        return RolloutBatch(
            states=states_t,
            actions=actions_t,
            logprobs=logprobs_t,
            returns=returns_t,
            advantages=adv_t,
        )


# ---------------------------------------------------------------------
# 4) Agente PPO
# ---------------------------------------------------------------------
class PPOAgent:
    """Agente PPO con actor-crítico LSTM."""

    def __init__(
        self,
        num_features: int,
        action_dim: int = 6,
        config: ProjectConfig = CONFIG,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.action_dim = int(action_dim)

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

        self.actor = ActorLSTM(
            num_features=num_features,
            action_dim=self.action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.critic = CriticLSTM(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.ppo.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.ppo.critic_lr)

        # Hiperparámetros PPO
        self.gamma = float(self.config.ppo.gamma)
        self.gae_lambda = float(self.config.ppo.gae_lambda)
        self.clip_eps = float(self.config.ppo.clip_eps)
        self.entropy_coef = float(self.config.ppo.entropy_coef)
        self.value_coef = float(self.config.ppo.value_coef)
        self.max_grad_norm = float(self.config.ppo.max_grad_norm)
        self.target_kl = float(self.config.ppo.target_kl)
        self.ppo_epochs = int(self.config.ppo.ppo_epochs)
        self.mini_batch_size = int(self.config.ppo.mini_batch_size)

        # std de política Gaussiana (constante global por dimensión)
        std_init = float(self.config.ppo.action_std_init)
        self.action_std = std_init
        self.action_std_min = float(self.config.ppo.action_std_min)
        self.action_std_decay = float(self.config.ppo.action_std_decay)

        self.rollout_buffer = RolloutBuffer()

    def decay_action_std(self) -> None:
        self.action_std = max(self.action_std_min, self.action_std * self.action_std_decay)

    def _get_dist(self, states: torch.Tensor) -> Normal:
        mu = self.actor(states)  # (B, A)
        std = torch.full_like(mu, fill_value=self.action_std)
        dist = Normal(mu, std)
        return dist

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,       # (T, F) o (1,T,F)
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Retorna acción, logprob y value estimado para el estado."""
        self.actor.eval()
        self.critic.eval()

        state_np = state.astype(np.float32)
        if state_np.ndim == 2:
            state_np = np.expand_dims(state_np, axis=0)  # (1,T,F)

        state_t = torch.from_numpy(state_np).to(self.device)

        dist = self._get_dist(state_t)
        value_t = self.critic(state_t).squeeze(-1)  # (1,)

        if deterministic:
            action_t = dist.mean
        else:
            action_t = dist.sample()

        #action_t = torch.clamp(action_t, -1.0, 1.0)
        logprob_t = dist.log_prob(action_t).sum(dim=-1)  # (1,)

        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        logprob = float(logprob_t.item())
        value = float(value_t.item())

        return action, logprob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.rollout_buffer.add(state, action, logprob, reward, done, value)

    def compute_gae_and_returns(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula ventajas GAE y retornos."""
        rewards = np.asarray(self.rollout_buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.rollout_buffer.dones, dtype=np.float32)
        values = np.asarray(self.rollout_buffer.values, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = float(last_value)
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = float(values[t + 1])
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values

        # normalización de ventajas (estándar PPO)
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return returns, advantages

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Actualiza actor y critic con PPO."""
        if len(self.rollout_buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        self.actor.train()
        self.critic.train()

        returns_np, adv_np = self.compute_gae_and_returns(last_value=last_value)
        batch = self.rollout_buffer.build_tensors(returns_np, adv_np, self.device)

        n_samples = batch.states.shape[0]
        idxs = np.arange(n_samples)

        actor_losses: List[float] = []
        critic_losses: List[float] = []
        entropies: List[float] = []
        kls: List[float] = []

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)

            for start in range(0, n_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = idxs[start:end]

                states_mb = batch.states[mb_idx]
                actions_mb = batch.actions[mb_idx]
                old_logprobs_mb = batch.logprobs[mb_idx]
                returns_mb = batch.returns[mb_idx]
                adv_mb = batch.advantages[mb_idx]

                dist = self._get_dist(states_mb)
                new_logprobs = dist.log_prob(actions_mb).sum(dim=-1)     # (B,)
                entropy = dist.entropy().sum(dim=-1).mean()

                values_pred = self.critic(states_mb).squeeze(-1)         # (B,)

                ratio = torch.exp(new_logprobs - old_logprobs_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(values_pred, returns_mb)

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logprobs_mb - new_logprobs).mean().item()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))
                kls.append(float(approx_kl))

            # early stop por KL
            mean_kl = float(np.mean(kls)) if kls else 0.0
            if mean_kl > self.target_kl:
                break

        self.rollout_buffer.clear()
        self.decay_action_std()

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "action_std": float(self.action_std),
        }