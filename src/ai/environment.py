from ai.model import CheckersModel


class CheckersEnv:
    def __init__(self):
        super().__init__()
        self.model = CheckersModel()

    def step(self, action: int):
        move = self.action_to_move(action)
        self.model.apply_move(move)
        done = self.model.is_ended()
        # Assume player 1 is the agent
        if done:
            if self.model.check_winner() == 1:
                reward = 10
            elif self.model.check_winner() == -1:
                reward = -10
            else:
                reward = 0
        else:
            reward = -0.01

        reward = 1 if done else 0
        state = self.model.get_state()
        valid_actions = self.get_valid_actions()
        return state, reward, done, valid_actions

    def get_valid_actions(self):
        valid_moves = self.model.get_valid_moves()
        return [self.move_to_action(move) for move in valid_moves]

    def reset(self):
        self.model.reset()
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
        (start_x, start_y), (end_x, end_y) = move
        start = start_y * 8 + start_x
        end = end_y * 8 + end_x
        return start * 64 + end

    def action_to_move(self, action: int):
        start, end = divmod(action, 64)
        start_y, start_x = divmod(start, 8)
        end_y, end_x = divmod(end, 8)
        return (start_x, start_y), (end_x, end_y)
