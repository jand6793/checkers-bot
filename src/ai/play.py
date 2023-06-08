from pathlib import Path

from agent import DeepQAgent
from environment import CheckersEnv


def play(agent: DeepQAgent, env: CheckersEnv):
    state = env.reset()
    valid_actions = env.get_valid_actions()
    done = False

    while not done:
        action = agent.act(state, valid_actions)
        print(f"Agent action: {env.action_to_move(action)}")
        next_state, reward, done, next_valid_actions = env.step(action)
        state, valid_actions = next_state, next_valid_actions

        while True:
            user_input = input("Enter action: ")
            user_nums = [int(x) for x in user_input.split(" ")]
            user_move = ((user_nums[0], user_nums[1]), (user_nums[2], user_nums[3]))
            user_action = env.move_to_action(user_move)
            valid_actions = env.get_valid_actions()
            if user_action in valid_actions:
                break
            print("Invalid action!")
        next_state, reward, done, next_valid_actions = env.step(user_action)
        state, valid_actions = next_state, next_valid_actions

    if reward == 1:
        print("Agent won!")
    elif reward == -1:
        print("Agent lost!")
    else:
        print("Draw!")


env = CheckersEnv()
agent = DeepQAgent(32, 4096)
agent.load(Path.cwd() / "agent_weights_1000.pt")
play(agent, env)
