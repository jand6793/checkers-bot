import sys
from pathlib import Path

import numpy as np
import torch

# Set random seeds
np.random.seed(34)
torch.manual_seed(34)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(34)
sys.path.append((Path.cwd() / "src").as_posix())

from agent import DeepQAgent
from environment import CheckersEnv


# Train the agent by playing against itself
def train(agent: DeepQAgent, env: CheckersEnv, episodes: int):
    p1_wins = 0
    p2_wins = 0
    draws = 0

    for episode in range(episodes):
        state = env.reset()
        valid_actions = env.get_valid_actions()
        done = False

        count = 0
        # Play the game until it's over or 200 moves have been made
        while not done and count < 200:
            count += 1
            action = agent.act(state, valid_actions)
            # Decay epsilon from 1 to 0.01 over the course of training
            agent.set_epsilon(max(1 - (episode / episodes), 0.01))
            # Move to the next state and get the reward
            next_state, reward, done, next_valid_actions = env.step(action)
            # Learn from the experience
            agent.learn(state, action, reward, next_state, done)

            # If the agent jumped, give a negative reward to the opponent's last move
            if env.replay[0].is_jump:
                opp_last_item_index = env.get_opp_last_item_index()
                opp_last_item = env.replay[opp_last_item_index]
                subsequent_item = env.replay[opp_last_item_index + 1]
                new_base_reward = -(opp_last_item_index + 1)
                # If the opponent won, give a reward of -10 in addition to each piece captured
                new_reward = (
                    new_base_reward - 10
                    if done and env.model.check_winner() != 0
                    else new_base_reward
                )
                agent.learn(
                    opp_last_item.state,
                    opp_last_item.action,
                    new_reward,
                    subsequent_item.state,
                    done,
                )

            state, valid_actions = next_state, next_valid_actions
        # If the game is over, record game stats
        if done:
            winner = env.model.check_winner()
            if winner == 1:
                p1_wins += 1
            elif winner == -1:
                p2_wins += 1
            else:
                draws += 1

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode+1}/{episodes} finished. P1 Wins: {p1_wins}, P2 Wins: {p2_wins}, Draws: {draws}"
            )


state_size = 32  # The number of availiable squares
action_size = 170  # The total number of actions
episodes = 10_000  # The number of episodes to train for

env = CheckersEnv()
agent = DeepQAgent(state_size, action_size)

train(agent, env, episodes)

weights_path = (
    Path()
    / f"weights_{episodes}ep_{agent.gamma}gamma_{agent.main_network.num_layers}layers-0.pt"
)

print(f'Training finished. Saving agent weights "{weights_path.name}"')
agent.save(weights_path)
