import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, padding='same') # 28*28*32
    self.conv2 = nn.Conv2d(32, 64, 3, padding='same') # 28*28*64
    self.maxpool = nn.MaxPool2d(2, 2) # 14*14*64
    self.dropout = nn.Dropout2d()
    self.fc1 = nn.Linear(14 * 14 * 64, 128)
    self.fc2 = nn.Linear(128, 10)
  
  def forward(self, input):
    x = F.relu(self.conv1(input))
    x = F.relu(self.conv2(x))
    x = self.maxpool(x)
    x = self.dropout(x)
    x = x.view(-1, 14 * 14 * 64)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return x