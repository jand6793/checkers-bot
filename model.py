from typing import Union

import numpy as np


# Game board representation and utility functions
class CheckersModel:
    def __init__(self):
        self._board = np.zeros((8, 8), dtype=int)
        self.reset()
        self._valid_moves = self._calc_valid_moves()
        self._current_player = 1
        self.num_p1_pieces = 12
        self.num_p2_pieces = 12

    # Check if the given position is within the board boundaries
    def _in_bounds(self, x: int, y: int):
        return 0 <= x < 8 and 0 <= y < 8

    # Recursively find all possible jumps for the give_n position
    def _find_jumps(
        self, x: int, y: int, visited: Union[set[tuple[int, int]], None] = None
    ):
        visited = visited or set()
        visited.add((x, y))

        jumps: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for dx, dy in self._get_directions(self._board[y, x]):
            new_x, new_y = x + dx, y + dy
            jump_x, jump_y = new_x + dx, new_y + dy
            if (
                self._in_bounds(new_x, new_y)
                and self._in_bounds(jump_x, jump_y)
                and self._board[new_y, new_x] == -self._current_player
                and self._board[jump_y, jump_x] == 0
                and (jump_x, jump_y) not in visited
            ):
                visited.add((jump_x, jump_y))
                jumps.append(((x, y), (jump_x, jump_y)))
                jumps.extend(self._find_jumps(jump_x, jump_y, visited))
        return jumps

    # Get a list of valid moves (jumps or regular moves) for the current player
    def _calc_valid_moves(self):
        valid_moves: list[tuple[tuple[int, int], tuple[int, int]]] = []
        jumps: list[tuple[tuple[int, int], tuple[int, int]]] = []

        # Find positions of all pieces for the current player
        piece_positions = np.argwhere(self._board * self._current_player > 0)

        for y, x in piece_positions:
            for dx, dy in self._get_directions(self._board[y, x]):
                new_x, new_y = x + dx, y + dy
                if self._in_bounds(new_x, new_y) and self._board[new_y, new_x] == 0:
                    valid_moves.append(((x, y), (new_x, new_y)))
            jumps.extend(self._find_jumps(x, y))
        return jumps or valid_moves

    # Get the directions a piece can move in
    def _get_directions(self, piece: int):
        return (
            [
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
            if abs(piece) == 2
            else [
                (-1, 1),
                (1, 1),
            ]
            if piece == 1
            else [
                (-1, -1),
                (1, -1),
            ]
        )

    # Apply a move on the board
    def apply_move(self, move: tuple[tuple[int, int], tuple[int, int]]):
        (start_x, start_y), (end_x, end_y) = move
        self._board[end_y, end_x] = self._board[start_y, start_x]
        self._board[start_y, start_x] = 0

        # Check if a jump was made, and remove the captured piece
        if abs(start_x - end_x) == 2:
            captured_x, captured_y = (start_x + end_x) // 2, (start_y + end_y) // 2
            self._board[captured_y, captured_x] = 0

        # Check if a piece became a king
        if (end_y == 0 and self._board[end_y, end_x] == -1) or (
            end_y == 7 and self._board[end_y, end_x] == 1
        ):
            self._board[end_y, end_x] *= 2

        if self._is_jump(move):
            if self._current_player == 1:
                self.num_p2_pieces -= 1
            else:
                self.num_p1_pieces -= 1

        # Switch to the other player
        self._current_player = -self._current_player

        self._valid_moves = self._calc_valid_moves()

    def _is_jump(self, move: tuple[tuple[int, int], tuple[int, int]]):
        return abs(move[0][0] - move[1][0]) == 2

    # Check if there's a winner (no more pieces for one player)
    def check_winner(self):
        if self.num_p1_pieces == 0:
            return -1
        elif self.num_p2_pieces == 0:
            return 1
        else:
            return 0

    def is_ended(self):
        return bool(self.check_winner()) or not self._calc_valid_moves()

    # Reset the game board to the initial state
    def reset(self):
        self._board = np.zeros((8, 8), dtype=int)
        self._board[:3:2, 1::2] = 1
        self._board[1, ::2] = 1
        self._board[5:8:2, ::2] = -1
        self._board[6, 1::2] = -1
        self._current_player = 1
        self.num_p1_pieces = 12
        self.num_p2_pieces = 12

    def get_state(self):
        flattened_board = self._board.flatten()
        return [
            flattened_board[i * 8 + j]
            for i in range(8)
            for j in range((i + 1) % 2, 8, 2)
        ]

    def get_valid_moves(self):
        return self._valid_moves
    
    def get_current_player(self):
        return self._current_player


if __name__ == "__main__":
    checkers = CheckersModel()
    moves = checkers.get_valid_moves()
    state = checkers.get_state()
    None
