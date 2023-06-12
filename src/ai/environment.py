import itertools
from collections import deque
from dataclasses import dataclass

from ai.model import CheckersModel


@dataclass
class ReplayItem:
    player: int
    state: list[int]
    action: int
    is_jump: bool


class CheckersEnv:
    """A wrapper for the CheckersModel that provides a gym-like interface."""
    def __init__(self):
        self.model = CheckersModel()
        self._move_to_action = self._create_move_to_action()
        self._action_to_move = self._create_action_to_move()
        self.replay: deque[ReplayItem] = deque()
        self.reset()

    # Apply the given action to the model and return the new state,
    # reward, done, and valid actions
    def step(self, action: int):
        move = self.action_to_move(action)
        current_player = self.model.get_current_player()
        pre_state = self.model.get_state()
        self.model.apply_move(move)

        winner = self.model.check_winner()
        # Reward of 10 for winning moves, 1 for jumps, 0 or -0.01 for regular moves
        if winner:
            reward = 10
        elif self.model.is_jump(move):
            reward = 1
        else:
            reward = 0
            # reward = -0.01
        post_state = self.model.get_state()
        valid_actions = self.get_valid_actions()

        # Add the current move to the replay buffer
        item = ReplayItem(current_player, pre_state, action, self.model.is_jump(move))
        self.replay.appendleft(item)
        done = self.model.is_ended()
        return post_state, reward, done, valid_actions

    # Return the valid actions for the current player
    def get_valid_actions(self):
        valid_moves = self.model.get_valid_moves()
        return [self.move_to_action(move) for move in valid_moves]

    # Reset the model and replay buffer
    def reset(self):
        self.model.reset()
        self.replay.clear()
        self.replay.appendleft(ReplayItem(0, self.model.get_state(), -1, False))
        return self.model.get_state()

    # Render the board state
    def render(self, mode="human"):
        print(self.model._board)  # This could be fancier, but for debugging it's okay.

    # Convert a move to an action
    # An action is an integer mapping to a move
    def move_to_action(self, move: tuple[tuple[int, int], tuple[int, int]]):
        return self._move_to_action[move]

    # Convert an action to a move
    def action_to_move(self, action: int):
        return self._action_to_move[action]

    # Create a mapping from moves to actions
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

    # Create a mapping from actions to moves
    def _create_action_to_move(self):
        return {v: k for k, v in self._move_to_action.items()}

    # Return the number of actions
    def get_action_space(self):
        return len(self._move_to_action)
    
    # Check if the given action is a jump
    def action_is_jump(self, action: int):
        return self.model.is_jump(self.action_to_move(action))

    # Get the index of the last move made by the opponent (opposite of the current player)
    def get_opp_last_item_index(self):
        opponent_player = -self.model.get_current_player()
        return next(
            (
                index
                for index, item in enumerate(self.replay)
                if item.player == opponent_player
            ),
            None,
        )


if __name__ == "__main__":
    env = CheckersEnv()

    valid_actions = env.get_valid_actions()
    next_state, reward, done, next_valid_actions = env.step(valid_actions[0])
    None
