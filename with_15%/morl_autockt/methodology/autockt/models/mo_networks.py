

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class MO_PPO_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, hidden_dim=64, layer_num=2):
        super(MO_PPO_Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        
        self.input_layer = nn.Linear(state_dim + reward_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(layer_num):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        self.apply(init_weights)
        
    def forward(self, state, preference):
        x = torch.cat([state, preference], dim=-1)
        
        x = F.relu(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        action_logits = self.action_head(x)
        
        return F.softmax(action_logits, dim=-1)

class MO_PPO_Critic(nn.Module):
    def __init__(self, state_dim, reward_dim, hidden_dim=64, layer_num=2):
        super(MO_PPO_Critic, self).__init__()
        
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        
        self.input_layer = nn.Linear(state_dim + reward_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(layer_num):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.value_head = nn.Linear(hidden_dim, reward_dim)
        
        self.apply(init_weights)
        
    def forward(self, state, preference):
        x = torch.cat([state, preference], dim=-1)
        
        x = F.relu(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        values = self.value_head(x)
        
        return values

class MO_DQN(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, hidden_dim=64, layer_num=2):
        super(MO_DQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        
        self.input_layer = nn.Linear(state_dim + reward_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(layer_num):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.q_head = nn.Linear(hidden_dim, action_dim * reward_dim)
        
        self.apply(init_weights)
        
    def forward(self, state, preference):
        x = torch.cat([state, preference], dim=-1)
        
        x = F.relu(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        q_values = self.q_head(x)
        q_values = q_values.view(-1, self.action_dim, self.reward_dim)
        
        return q_values

class PreferenceNetwork(nn.Module):
    def __init__(self, state_dim, reward_dim, hidden_dim=32):
        super(PreferenceNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.hidden_dim = hidden_dim
        
        self.pref_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, reward_dim),
            nn.Softmax(dim=-1)  # Ensure preferences sum to 1
        )
        
        self.apply(init_weights)
        
    def forward(self, state):
        return self.pref_net(state)

class CircuitSpecificationPredictor(nn.Module):
    def __init__(self, param_dim, spec_dim, hidden_dim=64):
        super(CircuitSpecificationPredictor, self).__init__()
        
        self.param_dim = param_dim
        self.spec_dim = spec_dim
        self.hidden_dim = hidden_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, spec_dim)
        )
        
        self.apply(init_weights)
        
    def forward(self, params):
        return self.predictor(params)
