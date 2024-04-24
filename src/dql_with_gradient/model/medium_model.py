import torch
from . import base_model
# import base_model

class GDQL_m(base_model.NNAdapter):
    def __init__(self, lr:float=0.001) -> None:
        '''
        This model take the board with size (2, 4, 8) and output the value of the action in the board
        Input:
            board: size (2, 4, 8) which is the state of the game first channel is board and second is moving player
        Output:
            value: size (1) which is the value of the action in the board
        '''
        super(GDQL_m, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 16, 3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1, stride=1)

        self.fc1 = torch.nn.Linear(64*4*8, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 1024)
        self.fc4 = torch.nn.Linear(1024, 512)
        self.fc5 = torch.nn.Linear(512, 1)

        self.relu = torch.nn.GELU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 64*4*8)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))

        x = self.fc5(x)

        return x
    
    def board2input(self, board, color, action):
        '''
        convert board dict to input of model
        '''
        model_input = torch.zeros(2, 32)
        model_input[0] = self.board2matrix(board, color).reshape(32)
        for seq in action:
            model_input[1][seq] = 1
        
        return model_input.reshape(1, 2, 4, 8)


if __name__ == '__main__':
    model = GDQL_m()
    import torchsummary as summary
    summary.summary(model, (2, 4, 8))