import itertools
import sys
from pathlib import Path

sys.path.append((Path.cwd() / "src").as_posix())

from ai.model import CheckersModel


class CheckersEnv:
    def __init__(self):
        self.model = CheckersModel()
        self._move_to_action = self._create_move_to_action()
        self._action_to_move = self._create_action_to_move()
        self._prev_state: list[int] = []
        self._prev_action: int = 0
        self._temp_action: int = 0
        self.reset()

    def step(self, action: int):
        self._prev_state = self.model.get_state()
        self._prev_action = self._temp_action
        self._temp_action = action
        move = self.action_to_move(action)
        self.model.apply_move(move)
        
        done = self.model.is_ended()
        winner = self.model.check_winner()
        reward = 10 if winner else -0.01
        state = self.model.get_state()
        valid_actions = self.get_valid_actions()
        return state, reward, done, valid_actions

    def get_valid_actions(self):
        valid_moves = self.model.get_valid_moves()
        return [self.move_to_action(move) for move in valid_moves]

    def reset(self):
        self.model.reset()
        self._prev_state = []
        self._prev_action = 0
        self._temp_action = 0
        return self.model.get_state()

    def render(self, mode="human"):
        print(self.model._board)  # This could be fancier, but for debugging it's okay.

    def get_reward(self):
        winner = self.model.check_winner()
        if not winner:
            return 0
        elif winner == 1:
            return 1
        else:
            return -1

    def move_to_action(self, move: tuple[tuple[int, int], tuple[int, int]]):
        return self._move_to_action[move]

    def action_to_move(self, action: int):
        return self._action_to_move[action]

    def _create_move_to_action(self):
        action_mapping: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        action_index = 0
        # change these depending on your coordinate systems
        dydx = [(-1, -1), (1, -1), (-1, 1), (1, 1), (-2, -2), (2, -2), (-2, 2), (2, 2)]

        # For each square on the board
        for i, j in itertools.product(range(8), range(8)):
            if (i + j) % 2 == 1:  # Check if the square is dark
                # For each possible action from this square
                for dx, dy in dydx:
                    new_i = i + dx
                    new_j = j + dy

                    # Check if the new square is valid and dark
                    if 0 <= new_i < 8 and 0 <= new_j < 8 and (new_i + new_j) % 2 == 1:
                        # If the action is valid, add it to the mapping
                        action_mapping[((i, j), (new_i, new_j))] = action_index
                        action_index += 1

        return action_mapping

    def _create_action_to_move(self):
        return {v: k for k, v in self._move_to_action.items()}

    def get_action_space(self):
        return len(self._move_to_action)

    def action_is_jump(self, action: int):
        return self.model.is_jump(self.action_to_move(action))

    def get_prev_state(self):
        return self._prev_state

    def get_prev_action(self):
        return self._prev_action


if __name__ == "__main__":
    env = CheckersEnv()

    valid_actions = env.get_valid_actions()
    next_state, reward, done, next_valid_actions = env.step(valid_actions[0])
    None
