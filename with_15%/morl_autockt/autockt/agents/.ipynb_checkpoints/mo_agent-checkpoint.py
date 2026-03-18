"""
Multi-Objective Reinforcement Learning Agent for AutoCkt
Based on PD-MORL framework adapted for circuit design optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from autockt.models.mo_networks import MO_PPO_Actor, MO_PPO_Critic, MO_DQN, PreferenceNetwork

MOExperience = namedtuple('MOExperience', 
                         ['state', 'action', 'reward_vector', 'next_state', 'done', 'preference'])

class ExperienceReplayBuffer_MO:
    """Multi-objective experience replay buffer"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward_vector, next_state, done, preference):
        experience = MOExperience(state, action, reward_vector, next_state, done, preference)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
        
    def __len__(self):
        return len(self.buffer)

class MO_PPO_Agent:
    """
    Multi-Objective PPO Agent for AutoCkt
    Implements preference-driven multi-objective optimization
    """
    
    def __init__(self, state_dim, action_dim, reward_dim, device='cpu', lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim  
        self.reward_dim = reward_dim
        self.device = device
        
        self.actor = MO_PPO_Actor(state_dim, action_dim, reward_dim).to(device)
        self.critic = MO_PPO_Critic(state_dim, reward_dim).to(device)
        self.preference_net = PreferenceNetwork(state_dim, reward_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.pref_optimizer = optim.Adam(self.preference_net.parameters(), lr=lr)
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        
        self.current_preference = None
        self.preference_history = []
        
    def set_preference(self, preference):
        """Set the current preference vector"""
        self.current_preference = torch.FloatTensor(preference).to(self.device)
        
    def generate_preference(self, state):
        """Generate preference vector based on current state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            preference = self.preference_net(state_tensor).squeeze(0)
        return preference.cpu().numpy()
        
    def select_action(self, state, preference=None):
        """Select action using current policy"""
        if preference is None:
            preference = self.current_preference
        else:
            preference = torch.FloatTensor(preference).to(self.device)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor, preference.unsqueeze(0))
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
        return action.cpu().numpy()[0]
        
    def compute_advantages(self, rewards, values, dones):
        """Compute GAE advantages for multi-objective rewards"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            if len(rewards[t].shape) > 0:  # Vector reward
                scalar_reward = np.dot(rewards[t], self.current_preference.cpu().numpy())
                scalar_value = np.dot(values[t], self.current_preference.cpu().numpy())
                scalar_next_value = np.dot(next_value, self.current_preference.cpu().numpy())
            else:  # Scalar reward (backward compatibility)
                scalar_reward = rewards[t]
                scalar_value = values[t]
                scalar_next_value = next_value
                
            delta = scalar_reward + self.gamma * scalar_next_value * (1 - dones[t]) - scalar_value
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return advantages
        
    def update(self, states, actions, rewards, next_states, dones, preferences):
        """Update actor, critic, and preference networks"""
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        preferences_tensor = torch.FloatTensor(preferences).to(self.device)
        
        values = self.critic(states_tensor, preferences_tensor)
        
        advantages = self.compute_advantages(rewards, values.detach().cpu().numpy(), dones)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        action_probs = self.actor(states_tensor, preferences_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action_log_probs = action_dist.log_prob(actions_tensor)
        
        with torch.no_grad():
            old_action_log_probs = action_log_probs.clone()
            
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
        actor_loss = -torch.min(surr1, surr2).mean()
        
        value_targets = []
        for i, reward in enumerate(rewards):
            if len(reward.shape) > 0:  # Vector reward
                target = reward + self.gamma * values[i+1] if i < len(rewards)-1 else reward
            else:  # Scalar reward
                target = np.array([reward] * self.reward_dim)  # Convert to vector
            value_targets.append(target)
            
        value_targets_tensor = torch.FloatTensor(value_targets).to(self.device)
        critic_loss = F.mse_loss(values[:-1], value_targets_tensor)
        
        entropy_loss = -action_dist.entropy().mean()
        
        total_loss = actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }

class MO_DQN_Agent:
    """
    Multi-Objective DQN Agent for AutoCkt (alternative to PPO)
    """
    
    def __init__(self, state_dim, action_dim, reward_dim, device='cpu', lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.device = device
        
        self.q_network = MO_DQN(state_dim, action_dim, reward_dim).to(device)
        self.target_network = MO_DQN(state_dim, action_dim, reward_dim).to(device)
        self.preference_net = PreferenceNetwork(state_dim, reward_dim).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.pref_optimizer = optim.Adam(self.preference_net.parameters(), lr=lr)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.target_update_freq = 100
        self.update_count = 0
        
        self.memory = ExperienceReplayBuffer_MO()
        
        self.current_preference = None
        
    def set_preference(self, preference):
        """Set current preference vector"""
        self.current_preference = torch.FloatTensor(preference).to(self.device)
        
    def select_action(self, state, preference=None, epsilon=None):
        """Select action using epsilon-greedy policy"""
        if preference is None:
            preference = self.current_preference
        else:
            preference = torch.FloatTensor(preference).to(self.device)
            
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor, preference.unsqueeze(0))
        
        scalarized_q = torch.sum(q_values * preference.unsqueeze(0).unsqueeze(1), dim=2)
        
        return scalarized_q.argmax().item()
        
    def store_experience(self, state, action, reward_vector, next_state, done, preference):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward_vector, next_state, done, preference)
        
    def update(self, batch_size=32):
        """Update Q-network using batch of experiences"""
        if len(self.memory) < batch_size:
            return {}
            
        batch = self.memory.sample(batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward_vector for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        preferences = torch.FloatTensor([e.preference for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states, preferences)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.reward_dim)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states, preferences)
            next_actions = torch.sum(self.q_network(next_states, preferences) * preferences.unsqueeze(1), dim=2).argmax(1)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.reward_dim)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones).unsqueeze(1)
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
