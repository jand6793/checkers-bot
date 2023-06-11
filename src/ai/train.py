import sys
from pathlib import Path

sys.path.append((Path.cwd() / "src").as_posix())

from agent import DeepQAgent
from environment import CheckersEnv


def self_play_train(agent: DeepQAgent, env: CheckersEnv, episodes: int):
    p1_wins = 0
    p2_wins = 0
    draws = 0

    for episode in range(episodes):
        state = env.reset()
        valid_actions = env.get_valid_actions()
        done = False

        count = 0
        while not done and count < 200:
            count += 1
            action = agent.act(state, valid_actions)
            next_state, reward, done, next_valid_actions = env.step(action)
            if done and env.model.check_winner() != 0:
                agent.learn(
                    env.get_prev_state(), env.get_prev_action(), -10, next_state, done
                )

            agent.learn(state, action, reward, next_state, done)
            state, valid_actions = next_state, next_valid_actions

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


# Set parameters
state_size = 32  # The number of squares in the checkers board
action_size = 170  # The total number of actions in the checkers game (each square can be a starting and ending point for a move)
episodes = 100_000  # The number of episodes to train for

# Create environment and agent
env = CheckersEnv()
agent = DeepQAgent(state_size, action_size)

# Train the agent
self_play_train(agent, env, episodes)
weights_path = Path() / f"weights_{episodes}ep_{agent.epsilon_decay}epsilDec_{agent.gamma}gamma_{agent.network.num_layers}layers.pt"
print(f'Training finished. Saving agent weights "{weights_path.name}"')
agent.save(weights_path)


None
