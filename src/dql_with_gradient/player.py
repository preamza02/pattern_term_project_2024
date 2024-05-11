import os
import sys
import time
import numpy as np
from collections import deque
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
while 'src' in current_dir.split(os.sep):
    current_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(current_dir, 'gym-checkers-for-thai'))

from checkers.agents.alpha_beta import Player, material_value, board_value
from model.small_model import GDQL
import torch


class DeepLearningPlayer(Player):

    experience = deque(maxlen=1000)

    def __init__(self, 
                 color: str, 
                 seed: int =None, 
                 model: GDQL = None,
                 epsilon: float = 0.8,
                 epsilon_decay: float = 0.80,
                 epsilon_min: float = 0.01,
                 win_reward: int = 100,
                 lose_reward: int = -50,
                 draw_reward: int = -20
                 ):
        super().__init__(color, seed)

        self.model = GDQL() if model is None else model

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.__win_reward = win_reward
        self.__lose_reward = lose_reward
        self.__draw_reward = draw_reward


    def next_move(self, board, last_moved_piece):
        self.model.eval()
        state = (board, self.color, last_moved_piece)
        self.simulator.restore_state(state)

        move_value = []
        for move in self.simulator.legal_moves():
            model_input = self.model.board2input(board, self.color, move)
            value = self.model(model_input).detach().numpy()[0]
            move_value.append((value, move))
    
        move_value.sort(key=lambda x: x[0], reverse=True)
        best_move = move_value[0][1]
        random_move = move_value[self.random.randint(len(move_value))][1]
        selected_move = best_move if self.random.rand() > self.epsilon else random_move

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # update experience
        if len(DeepLearningPlayer.experience) > 0 and DeepLearningPlayer.experience[-1][3] is None:
            DeepLearningPlayer.experience[-1][3] = copy.deepcopy(state)
        DeepLearningPlayer.experience.append([copy.deepcopy(state),
                                      selected_move, 
                                      self.board_reward(board, self.simulator.board),
                                      None])

        return selected_move

    def board_reward(self, board, next_board):
        board_reward = material_value(0.86, 0.545, board[self.color])
        board_reward += board_value("black", 0.87, 0.55, board[self.color])
        board_reward -= material_value(0.86, 0.545, board["white" if self.color == "black" else "black"])
        board_reward -= board_value("white", 0.87, 0.55, board["white" if self.color == "black" else "black"])

        return board_reward
    
    def get_max_q(self, board):
        self.model.eval()
        max_q = -np.inf
        state = (board, self.color, None)
        self.simulator.restore_state(state)
        for move in self.simulator.legal_moves():
            model_input = self.model.board2input(board, self.color, move)
            value = self.model(model_input).detach().numpy()[0]
            max_q = max(max_q, value)
        return max_q[0]


    def set_win(self):
        DeepLearningPlayer.experience[-1][2] = self.__win_reward
        DeepLearningPlayer.experience[-1][3] = None
    
    def set_lose(self):
        DeepLearningPlayer.experience[-1][2] = self.__lose_reward
        DeepLearningPlayer.experience[-1][3] = None

    def set_draw(self):
        DeepLearningPlayer.experience[-1][2] = self.__draw_reward
        DeepLearningPlayer.experience[-1][3] = None

    
if __name__ == '__main__':
    from checkers.game import Checkers
    from checkers.agents.baselines import play_a_game, RandomPlayer

    print('round 1')
    ch = Checkers()

    black_player = DeepLearningPlayer("black", seed=0)
    # white_player = DeepLearningPlayer("white", seed=1)
    white_player = RandomPlayer("white", seed=1)

    winner = play_a_game(ch, black_player.next_move, white_player.next_move, 100, is_show_detail=False)
    # play_a_game(ch, keyboard_player_move, keyboard_player_move)
    print(winner)
    if winner == "black":
        black_player.set_win()
    elif winner == "white":
        black_player.set_lose()
    else:
        black_player.set_draw()

    for idx, x in enumerate(black_player.experience):
        print(idx)
        for y in x:
            print('\t', y)
