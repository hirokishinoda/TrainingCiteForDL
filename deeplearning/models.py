import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5) # 24*24*32
        self.pool1 = nn.MaxPool2d(2, 2) # 12*12*32
        self.conv2 = nn.Conv2d(32, 64, 3) # 10*10*64 
        self.pool2 = nn.MaxPool2d(2, 2) # 5*5*64
        self.linear1 = nn.Linear(5*5*64, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool2(F.relu(self.conv2(out)))
        out = out.view(-1, 5*5*64)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        return out
