from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        # A simple network with one hidden layer
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


class DeepQAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.95,
        learning_rate: float = 0.01,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.network = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.action_size = action_size

    def act(self, state: np.ndarray, valid_actions: list[int]):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.network.eval()
        with torch.no_grad():
            q_values = self.network(state)
        self.network.train()
        q_values = q_values.cpu().numpy()

        # Mask invalid actions
        masked_q_values = np.full(self.action_size, -np.inf)
        masked_q_values[valid_actions] = q_values[valid_actions]

        return np.argmax(masked_q_values)

    def learn(self, state, action, reward, next_state, done: bool):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(
            next_state, dtype=torch.float32, device=self.device
        )
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)

        if done:
            target_value = reward_tensor.to(device=self.device)
        else:
            with torch.no_grad():
                next_q_values = self.network(next_state_tensor)
            target_value = reward_tensor + self.gamma * torch.max(next_q_values)

        predicted_value = self.network(state_tensor)[action]

        loss = self.criterion(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: Path):
        torch.save(self.network.state_dict(), filepath)

    def load(self, filepath: Path):
        self.network.load_state_dict(torch.load(filepath))
