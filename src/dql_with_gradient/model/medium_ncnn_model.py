import torch
from torch import nn
from . import base_model
# import base_model

class GDQLnocnn_m(base_model.NNAdapter):
    def __init__(self, lr:float=0.001) -> None:
        '''
        This model take the board with size (2, 4, 8) and output the value of the action in the board
        Input:
            board: size (2, 4, 8) which is the state of the game first channel is board and second is moving player
        Output:
            value: size (1) which is the value of the action in the board
        '''
        super(GDQLnocnn_m, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # init weight and bias with 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.fc(x)
        return x
    
    def board2input(self, board, color, action):
        '''
        convert board dict to input of model
        '''
        model_input = torch.zeros(2, 32)
        model_input[0] = self.board2matrix(board, color).reshape(32)
        for seq in action:
            model_input[1][seq] = 1
        
        return model_input.reshape(1, 64)

if __name__ == '__main__':
    model = GDQLnocnn_m()
    import torchsummary as summary
    summary.summary(model, (64))