
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nets import Actor, CriticDual

class MADDPG_DualQ:
    def __init__(self, state_dim, act_dim, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, cfg.actor_hidden, act_dim).to(self.device)
        self.actor_targ = Actor(state_dim, cfg.actor_hidden, act_dim).to(self.device)
        self.critic = CriticDual(state_dim + act_dim, cfg.critic_hidden).to(self.device)
        self.critic_targ = CriticDual(state_dim + act_dim, cfg.critic_hidden).to(self.device)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.mse = nn.MSELoss()

    def act(self, s_np, noise_sigma):
        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).squeeze(0).detach().cpu().numpy()
        if noise_sigma > 0:
            a += np.random.normal(0, noise_sigma, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    def train_step(self, batch, gamma, tau):
        s, a, r, s2, d = [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch]
        with torch.no_grad():
            a2 = self.actor_targ(s2)
            noise = torch.clamp(torch.randn_like(a2) * self.cfg.target_noise_sigma,
                                -self.cfg.target_noise_clip, self.cfg.target_noise_clip)
            a2 = torch.clamp(a2 + noise, -1.0, 1.0)
            q1_t, q2_t = self.critic_targ(s2, a2)
            q_targ = torch.min(q1_t, q2_t)
            y = r + (1 - d) * gamma * q_targ

        q1, q2 = self.critic(s, a)
        loss_c = self.mse(q1, y) + self.mse(q2, y)
        self.critic_opt.zero_grad()
        loss_c.backward()
        self.critic_opt.step()

        a_online = self.actor(s)
        q1_val = self.critic.q1_only(s, a_online)
        loss_a = -q1_val.mean()
        self.actor_opt.zero_grad()
        loss_a.backward()
        self.actor_opt.step()

        for t, m in zip(self.actor_targ.parameters(), self.actor.parameters()):
            t.data.copy_(tau * m.data + (1 - tau) * t.data)
        for t, m in zip(self.critic_targ.parameters(), self.critic.parameters()):
            t.data.copy_(tau * m.data + (1 - tau) * t.data)