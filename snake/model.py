import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool, add_coords: bool = True):
        super(ConvBlock, self).__init__()
        self.residual = residual
        if residual:
            assert in_channels == out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv(x)
        out = F.relu(out)

        if self.residual:
            out += identity

        return out


def feedforward_block(input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU()
    )


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

class ConvAgent(nn.Module):
    """Implementation of baseline agent architecture from https://arxiv.org/pdf/1806.01830.pdf"""
    def __init__(self,
                 in_channels: int = 5,
                 num_initial_convs: int = 1,
                 num_residual_convs: int = 2,
                 num_feedforward: int = 1,
                 feedforward_dim: int = 32,
                 num_actions: int = 4,
                 conv_channels: int = 64,
                 num_heads: int = 1):
        super(ConvAgent, self).__init__()
        self.in_channels = in_channels
        self.num_initial_convs = num_initial_convs
        self.num_residual_convs = num_residual_convs
        self.num_feedforward = num_feedforward
        self.feedforward_dim = feedforward_dim
        self.conv_channels = conv_channels
        self.num_actions = num_actions
        self.num_heads = num_heads

        initial_convs = [ConvBlock(self.in_channels, self.conv_channels, residual=False), ]
        for _ in range(self.num_initial_convs - 1):
            initial_convs.append(ConvBlock(self.conv_channels, self.conv_channels, residual=False))

        self.initial_conv_blocks = nn.Sequential(*initial_convs)

        residual_convs = [ConvBlock(self.conv_channels, self.conv_channels, residual=True), ]
        for _ in range(self.num_residual_convs - 1):
            residual_convs.append(ConvBlock(self.conv_channels, self.conv_channels, residual=True))

        self.residual_conv_blocks = nn.Sequential(*residual_convs)

        feedforwards = [feedforward_block(self.conv_channels, self.feedforward_dim), ]
        for _ in range(self.num_feedforward - 1):
            feedforwards.append(feedforward_block(self.feedforward_dim, self.feedforward_dim))

        feedforwards.append(feedforward_block(self.feedforward_dim, self.num_actions))

        self.dense = nn.Sequential(*feedforwards)


    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.initial_conv_blocks(x.float())
        x = self.residual_conv_blocks(x)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.dense(x)
        return x