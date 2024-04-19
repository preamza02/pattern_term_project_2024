import torch

class DQN(torch.nn.Module):
    def __init__(self) -> None:
        '''
        This model use to predict value of each action in the state
        input is board which is size = (4x8,) and each element has its value which
            0 means empty,
            1 means man,
            2 means king,
            and negative value means opponent, positive value means player
        output is size (4x8x4x8,) which is the value of each action in the state
            to get action do
            1. reshape the output to (4x8, 4x8)
            and the value of O[i][j] is the value of action (i, j) in the state board
        '''
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(32, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 512)
        self.fc5 = torch.nn.Linear(512, 4*8*4*8)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def board2input(self, board, color):
        '''
        convert board dict to input of model
        '''
        model_input = torch.zeros(32)
        for current_color in ['white', 'black']:
            for current_type in ['men', 'kings']:
                for current_position in board[current_color][current_type]:
                    if current_type == "men":
                        model_input[current_position] = 1
                    elif current_type == "kings":
                        model_input[current_position] = 2
                    if current_color != color:
                        model_input[current_position] = -model_input[current_position]
        
        return model_input
    
if __name__ == '__main__':
    model = DQN()
    import torchsummary as summary
    summary.summary(model, (1, 32))