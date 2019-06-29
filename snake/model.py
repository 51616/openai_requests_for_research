import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch

class feedforward(nn.Module):
    def __init__(self, board_size=20, hidden_size=128):
        super(feedforward, self).__init__()
        self.hidden_size = hidden_size
        self.board_size = board_size
        self.dense1 = nn.Linear( (self.board_size**2) * 3, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, 4)

    def forward(self, x):
        #x = torch.from_numpy(x).float()
        x = x.view( -1, (self.board_size**2) * 4).float()
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

class convnet(nn.Module):
    def __init__(self, board_size=20, hidden_size=128, conv_channel = 64, in_channel=3):
        super(convnet, self).__init__()
        self.hidden_size = hidden_size
        self.board_size = board_size
        self.conv_channel = conv_channel
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel,self.conv_channel//2,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(self.conv_channel//2)
        self.conv2 = nn.Conv2d(self.conv_channel//2,self.conv_channel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(self.conv_channel)
        self.dense = nn.Linear(self.conv_channel*self.board_size**2, 4)

    def forward(self,x):
        x = x.view(-1,self.in_channel,self.board_size,self.board_size).float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dense(x.view(-1,self.conv_channel*self.board_size**2))
        # print('Model output:',x)
        return x

class convnet_duel(nn.Module):
    def __init__(self, board_size=20, hidden_size=128, conv_channel = 64, in_channel=3):
        super(convnet_duel, self).__init__()
        self.hidden_size = hidden_size
        self.board_size = board_size
        self.conv_channel = conv_channel
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel,self.conv_channel//2,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(self.conv_channel//2)
        self.conv2 = nn.Conv2d(self.conv_channel//2,self.conv_channel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(self.conv_channel)
        self.dense = nn.Linear(self.conv_channel*self.board_size**2, self.hidden_size)
        self.advantage_head = nn.Linear(self.hidden_size, 4)
        self.value_head = nn.Linear(self.hidden_size, 1)

    def forward(self,x):
        x = x.view(-1,self.in_channel,self.board_size,self.board_size).float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dense(x.view(-1,self.conv_channel*self.board_size**2))
        advantage = self.advantage_head(x)
        value = self.value_head(x)
        output = value + advantage - advantage.mean(1, keepdim=True) #Dueling DQN
        return output
