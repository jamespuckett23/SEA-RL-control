import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Hyperparameters
GAMMA = 0.99  # Discount factor
TAU = 0.005   # Soft update parameter
LR_ACTOR = 1e-4  # Learning rate for actor
LR_CRITIC = 1e-3  # Learning rate for critic
BUFFER_SIZE = int(1e6)  # Replay buffer size
BATCH_SIZE = 64  # Batch size for replay buffer
ACTION_LIMIT = np.pi  # Actions are bounded between -π and π

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit):
        super(Actor, self).__init__()
        self.action_limit = action_limit
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x)) * self.action_limit
        return action

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# Replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.float32(reward)  # Ensure reward is a scalar float
        next_state = np.array(next_state, dtype=np.float32)
        done = np.float32(done)  # Convert done to a float for consistency (0.0 or 1.0)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert directly to torch tensors with proper shapes
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        # rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        rewards = torch.FloatTensor([float(r) for r in rewards]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)  # Reshape dones to (batch_size, 1)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# class ReplayBuffer:
#     def __init__(self, buffer_size, batch_size):
#         self.buffer = []
#         self.buffer_size = buffer_size

#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#         if len(self.buffer) > self.buffer_size:
#             self.buffer.pop(0)

#     def sample(self, batch_size):
#         samples = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*samples)
        
#         # Convert them to tensors, ensuring they are in the right format
#         states = torch.FloatTensor(np.array(states))
#         actions = torch.FloatTensor(np.array(actions))
        
#         # Make sure rewards are simple scalars
#         rewards = torch.FloatTensor([float(r) for r in rewards]).unsqueeze(1)
#         next_states = torch.FloatTensor(np.array(next_states))
#         dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

#         return states, actions, rewards, next_states, dones
    
#     def __len__(self):
#         return len(self.buffer)

# DDPG agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_limit):
        self.actor = Actor(state_dim, action_dim, action_limit)
        self.actor_target = Actor(state_dim, action_dim, action_limit)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.noise = OUNoise(action_dim)
        
        self.soft_update(self.actor_target, self.actor, 1.0)
        self.soft_update(self.critic_target, self.critic, 1.0)
    
    def act(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        # action += self.noise.sample() * noise_scale  # Adding noise for exploration
        return action # np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)
    
    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) > BATCH_SIZE:
            self.learn()
    
    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Update Critic
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        q_targets = rewards + GAMMA * next_q_values * (1 - dones)
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, q_targets.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor, TAU)
        self.soft_update(self.critic_target, self.critic, TAU)
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Ornstein-Uhlenbeck noise for exploration
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
