import sys
from pathlib import Path

sys.path.append((Path.cwd() / "src").as_posix())

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSizePolicy,
    QWidget,
    QLabel,
    QVBoxLayout,
    QGridLayout,
)
from PySide6.QtGui import QColor, QPalette, QDrag, QPixmap
from PySide6.QtCore import Qt, QMimeData, QTimer, QObject, Signal

from ai.environment import CheckersEnv
from ai.agent import DeepQAgent

import numpy as np

TAN_SQUARE_PATH = Path.cwd() / "assets" / "tan_square.png"
BLACK_SQUARE_PATH = Path.cwd() / "assets" / "black_square.png"
TAN_PIECE_PATH = Path.cwd() / "assets" / "tan_piece.png"
BLACK_PIECE_PATH = Path.cwd() / "assets" / "black_piece.png"


class DraggableLabel(QLabel):
    def __init__(self, row, col, is_draggable):
        super().__init__()
        self.row = row
        self.col = col
        self.is_draggable = is_draggable

    def mousePressEvent(self, event):
        if self.is_draggable and event.button() == Qt.LeftButton:
            drag = QDrag(self)
            mime_data = QMimeData()
            drag.setMimeData(mime_data)

            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.exec()


class DroppableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        piece = event.source()
        start_pos = piece.col, piece.row
        end_pos = self.column, self.row
        piece.row, piece.col = end_pos  # Update piece position
        self.parentWidget().user_moved.emit((start_pos, end_pos))


class CheckersBoard(QWidget):
    user_moved = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.build_board()

    def build_board(self):
        for i in range(8):
            i_inverse = 7 - i % 8
            for j in range(8):
                if (i + j) % 2 == 0:
                    square = DroppableLabel()
                    pixmap = QPixmap(BLACK_SQUARE_PATH.as_posix())
                else:
                    square = QLabel()
                    pixmap = QPixmap(TAN_SQUARE_PATH.as_posix())
                square.setAutoFillBackground(True)
                square.setPixmap(
                    pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                square.row = i_inverse
                square.column = j
                square.setLayout(QVBoxLayout())  # Each square gets its own layout
                self.grid_layout.addWidget(square, i, j)

    def update_state(self, state: np.ndarray):
        for i in range(8):
            i_inverse = 7 - i % 8
            for j in range(8):
                widget = self.grid_layout.itemAtPosition(i, j).widget()
                # Clear the square and then add the new piece
                layout = widget.layout()
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

                piece = state[i_inverse, j]
                if piece in [1, -1]:
                    piece_label = DraggableLabel(i_inverse, j, True)
                    pixmap = QPixmap(
                        TAN_PIECE_PATH.as_posix()
                        if piece == 1
                        else BLACK_PIECE_PATH.as_posix()
                    ).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    piece_label.setPixmap(pixmap)
                    layout.addWidget(piece_label)


class CheckersGame(QObject):
    def __init__(self):
        super().__init__()

        self.agent = DeepQAgent(32, 4096)
        self.agent.load(Path.cwd() / "agent_weights_1000.pt")

        self.env = CheckersEnv()
        self.state = self.env.reset()
        self.valid_actions = self.env.get_valid_actions()

        self.window = QMainWindow()
        self.window.resize(800, 800)
        self.window.setCentralWidget(self.board)
        self.board = CheckersBoard()
        self.board.user_moved.connect(self.user_move)

        self.agent_turn()

    def agent_turn(self):
        action = self.agent.act(self.state, self.valid_actions)
        self.state, reward, done, self.valid_actions = self.env.step(action)
        self.board.update_state(self.env.model.get_expanded_state())

        if done:
            self.end_game(reward)

    def user_move(self, move):
        if move in self.env.model.get_valid_moves():
            action = self.env.move_to_action(move)
            if action in self.valid_actions:
                self.state, reward, done, self.valid_actions = self.env.step(action)
                self.board.update_state(self.env.model.get_expanded_state())

                if done:
                    self.end_game(reward)
                else:
                    QTimer.singleShot(0, self.agent_turn)

    def end_game(self, reward):
        if reward == 1:
            print("Agent won!")
        elif reward == -1:
            print("User won!")
        else:
            print("Draw!")

    def run(self):
        self.window.show()


app = QApplication(sys.argv)
# set the app resolution to 800x80
game = CheckersGame()
game.run()

sys.exit(app.exec())
