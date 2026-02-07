import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from utils.replay_buffer import ReplayBuffer
class DuelingQNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        features = self.feature(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.device = torch.device(device)
        self.action_size = action_size
        print(f"Initializing DQNAgent on {self.device}")
        
        self.q_net = DuelingQNet(state_size, action_size).to(self.device)
        self.target_net = DuelingQNet(state_size, action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.0005) # Slower LR for stability
        self.memory = ReplayBuffer(capacity=50000, device=self.device)
        self.update_target_net()

    def act(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_tensor).argmax().item()

    def train(self, batch_size):
        batch = self.memory.sample(batch_size)
        if batch is None:
            return None
            
        states, actions, rewards, next_states, dones = batch
        
        # Current Q values
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN Target Calculation
        with torch.no_grad():
            # Online net selects actions
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            # Target net evaluates those actions
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + 0.99 * next_q * (1 - dones)
        
        # MSE loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_target_net()
        print(f"Model loaded from {path}")
