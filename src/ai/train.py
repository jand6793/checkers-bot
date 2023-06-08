import sys
from pathlib import Path

sys.path.append((Path.cwd() / "src").as_posix())

from agent import DeepQAgent
from environment import CheckersEnv


def self_play_train(agent: DeepQAgent, env: CheckersEnv, episodes: int):
    win_count = 0
    loss_count = 0
    draw_count = 0

    for episode in range(episodes):
        state = env.reset()
        valid_actions = env.get_valid_actions()
        done = False

        count = 0
        while not done and count < 200:
            count += 1
            action = agent.act(state, valid_actions)
            next_state, reward, done, next_valid_actions = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state, valid_actions = next_state, next_valid_actions

        if reward == 1:
            win_count += 1
        elif reward == -1:
            loss_count += 1
        else:
            draw_count += 1

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode+1}/{episodes} finished. Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}"
            )


# Set parameters
state_size = 32  # The number of squares in the checkers board
action_size = 98  # The total number of actions in the checkers game (each square can be a starting and ending point for a move)
episodes = 1_000  # The number of episodes to train for

# Create environment and agent
env = CheckersEnv()
agent = DeepQAgent(state_size, action_size)

# Train the agent
self_play_train(agent, env, episodes)
agent.save(Path() / f"agent_weights_{episodes}_v2.pt")


None
