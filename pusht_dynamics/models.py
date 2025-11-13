import torch
import torch.nn as nn

def mlp(d_in, d_out, hidden=(256,256), act=nn.ReLU):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers += [nn.Linear(d, d_out)]
    return nn.Sequential(*layers)

class InverseDynamics(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden = (256, 256, 256)
        self.net = mlp(3*obs_dim, act_dim, hidden=hidden)

    def forward(self, o_prev, o_curr, o_next):
        x = torch.cat([o_prev, o_curr, o_next], dim=-1)
        return self.net(x)

class ForwardDynamics(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden = (256, 256, 256)
        self.net = mlp(obs_dim + act_dim, obs_dim, hidden=hidden)

    def forward(self, o_curr, a_t):
        x = torch.cat([o_curr, a_t], dim=-1)
        return self.net(x)
