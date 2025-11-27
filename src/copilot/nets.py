
import torch
import torch.nn as nn

def mlp(in_dim, hidden, out_dim, out_act=None):
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    if out_act:
        layers += [out_act]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden, act_dim):
        super().__init__()
        self.net = mlp(state_dim, hidden, act_dim)
    def forward(self, s):
        return torch.tanh(self.net(s))

class CriticDual(nn.Module):
    def __init__(self, joint_dim, hidden):
        super().__init__()
        self.q1 = mlp(joint_dim, hidden, 1)
        self.q2 = mlp(joint_dim, hidden, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)
    def q1_only(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x)