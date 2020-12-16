import torch
import torch.nn as nn

class R2D2(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.lstm1 = nn.LSTM(64*7*7, 512)
        self.lstm2 = nn.LSTM(512, n_actions)

    def forward(self, x, h=(None,None)):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = x.unsqueeze(0)
        x, (h1, _) = self.lstm1(x, h[0])
        x, (h2, _) = self.lstm2(x, h[1])
        return x.squeeze(0), (h1, h2)
