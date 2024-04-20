import os
import sys
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
while 'src' in current_dir.split(os.sep):
    current_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(current_dir, 'gym-checkers-for-thai'))

from checkers.agents import Player
from model import DQN
import torch

class GADQNPlayer(Player):
    def __init__(self, color: str, seed: int =1, model: DQN = None):
        super().__init__(color, seed)
        torch.set_grad_enabled(False)

        self.model = DQN() if model is None else model

        self._n_move_dicisions = 0
        self._n_moves = 0
        self._n_combo_kill = 0

    def next_move(self, board, last_moved_piece):
        state = (board, self.color, last_moved_piece)
        self.simulator.restore_state(state)

        q_value = self.get_q_values(board)

        move_values = [(q_value[move[0]][move[1]], move) for move in self.simulator.legal_moves()]
        move_values.sort(key=lambda x: -x[0])

        self._n_move_dicisions += len(move_values)
        self._n_moves += 1
        self._n_combo_kill += 1 if last_moved_piece is not None else 0

        return move_values[0][1]
    
    @property
    def n_move_dicisions(self):
        return self._n_move_dicisions
    
    @property
    def n_moves(self):
        return self._n_moves
    
    @property
    def n_combo_kill(self):
        return self._n_combo_kill

    def get_q_values(self, board):
        model_input = self.model.board2input(board, self.color)
        # model_input = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0)
        model_output = self.model(model_input).reshape(32, 32).detach().numpy()

        return model_output


if __name__ == '__main__':
    from checkers.game import Checkers
    from checkers.agents.baselines import play_a_game

    ch = Checkers()

    black_player = GADQNPlayer("black", seed=0)
    white_player = GADQNPlayer("white", seed=1)
    winner = play_a_game(ch, black_player.next_move, white_player.next_move, is_show_detail=True)
    # play_a_game(ch, keyboard_player_move, keyboard_player_move)
    print(winner)