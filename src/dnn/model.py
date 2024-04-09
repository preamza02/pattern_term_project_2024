import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, input_size=(3, 8, 4), output_size=(2, 32)):
        '''
        Input is tensor size 3*8*4 (3 channels, 8 rows, 4 columns)
            which first channel is 1 if it's adversarial piece, 0 otherwise
            second channel is 1 if it's king, 0 otherwise
            third channel is 1 if it's our piece, 0 otherwise
            e.g. if it's adv. king -> [1, 1, 0]
            or if it's our man -> [0, 0, 1]
        Output is tensor size 2*32*32 (2 channels, 32 rows, 32 columns)
            which one in 1st channel is from position of piece
            and one in 2nd channel is to position of piece
        '''
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*8*4, 512)
        self.fc2 = nn.Linear(512, 2*32)
        self.output = nn.Sequential(nn.Sigmoid(),
                                    nn.Unflatten(1, output_size))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)


    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)

        return x
    
    def predict(self, x):
        self.eval()
        x = self.forward(x)
        return x

if __name__ == "__main__":
    import torchsummary
    model = Model()
    input_size = (3, 8, 4)
    output_size = (2, 32)

    torchsummary.summary(model, input_size=input_size, device='cpu')

    x = torch.randn(1, *input_size)
    y = model(x)
    print(y)