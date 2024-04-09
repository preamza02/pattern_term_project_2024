from checkers.agents import Player
from checkers.game import Checkers

from model import Model

import torch

class CNNPlayer(Player):
    def __init__(self, color, seed=None):
        super(CNNPlayer, self).__init__(color, seed=seed)
        self.adv_color = 'black' if color == 'white' else 'white'
        self._model = Model()
        self._seed = seed
        self._histories = []

    def next_move(self, board, last_moved_piece):
        # TODO: integrate model to get next move
        state = board, self.color, last_moved_piece
        self.simulator.restore_state(state)
        legal_moves = self.simulator.legal_moves()

        current_board = self.state2input(state)
        model_output = self._model(current_board)
        next_move = self.output2sq(model_output, legal_moves)

        self._histories.append([current_board, model_output, next_move, None])
        return next_move

    def train(self):
        # TODO: train by self histroy and is_win
        losses = self.create_loss()
        self._model.train()
        self._model.optimizer.zero_grad()
        losses.backward()
        self._model.optimizer.step()
        self._histories = []

    
    def create_loss(self):
        # TODO: create dataset from self history

        # to calculate loss if is_win is true loss = 1-selected_output
        # if is_win is false loss = selected_output
        loss = 0
        for _, output, move, is_win in self._histories:
            if is_win is None:
                continue
            if is_win:
                loss += 1 - output[0][0][move[0]] * output[0][1][move[1]]
            else:
                loss += output[0][0][move[0]] * output[0][1][move[1]]

        return loss
        

    def state2input(self, state):
        board, color, _ = state
        input = torch.zeros(1, 3, 8, 4, dtype=torch.float32)

        for row in range(8):
            for col in range(4):
                current_sq = row * 4 + col
                if current_sq in board[self.adv_color]['kings'] or current_sq in board[self.adv_color]['men']:
                    input[0][0][row][col] = 1
                if current_sq in board[self.color]['kings'] or current_sq in board[self.color]['men']:
                    input[0][2][row][col] = 1    
                if current_sq in board[self.color]['kings'] or current_sq in board[self.adv_color]['kings']:
                    input[0][1][row][col] = 1

        return input
                

    def output2sq (self, output, legal_moves=None):
        '''
        Convert output from model to sequence
            output : 2*32*32 tensor from model output
            return : (from_sq, to_sq) (max value in 1st channel, max value in 2nd channel)
        '''
        output = output.clone().detach().reshape(2, 32)
        p_move = []
        for moves in legal_moves:
            from_sq, to_sq = moves
            p_move.append((output[0][from_sq]*output[1][to_sq], moves))
        
        p_move = sorted(p_move, key=lambda x: x[0], reverse=True)
        return p_move[0][1]

    
    @property
    def model(self):
        return self._model
    
    @property
    def history(self):
        return self._histories
    
    def win(self):
        '''
        Set all history which is_win is None to True
        '''
        self.is_win = True
        for i in range(len(self._histories)-1, -1, -1):
            if self._histories[i][-1] is None:
                self._histories[i][-1] = True
            else:
                break

    def lose(self):
        '''
        Set all history which is_win is None to False
        '''
        self.is_win = False
        for i in range(len(self._histories)-1, -1, -1):
            if self._histories[i][-1] is None:
                self._histories[i][-1] = False
            else:
                break
    
    def draw(self):
        '''
        Remove all history which is_win is None
        '''
        self.is_win = False
        self._histories = [h for h in self._histories if h[2] is not None]

    def save_model(self, path='CNNPlayer.pth'):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path='CNNPlayer.pth'):
        self._model.load_state_dict(torch.load(path))