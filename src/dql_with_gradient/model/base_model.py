import torch

class NNAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super(NNAdapter, self).__init__()

    def forward(self, x):
        raise NotImplementedError
    
    def board2matrix(self,
                     board,
                     our_color,
                     king_id = 2,
                     man_id = 1,):
        '''
        convert board dict to matrix of size (1, 4, 8)
        which represent the board of the game 
            our chacker and adversary checker recognize by positive and negative number
        '''
        matrix = torch.zeros(1, 32)
        for color in ['white', 'black']:
            for checker_type in ['men', 'kings']:
                for position in board[color][checker_type]:
                    if checker_type == 'men':
                        matrix[0][position] = man_id
                    else:
                        matrix[0][position] = king_id
                    if color != our_color:
                        matrix[0][position] = -matrix[0][position]

        return matrix.reshape(1, 4, 8)