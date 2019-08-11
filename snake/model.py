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


class ResBlock(nn.Module):
    def __init__(self, filters=64, kernel=3):
        super(ResBlock, self).__init__()
        if kernel % 2 != 1:
            raise ValueError('kernel must be odd, got %d' % kernel)
        # pad = int(np.floor(kernel/2))

        self.conv1 = nn.Conv2d(
            filters, filters, kernel_size=kernel, padding=0)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(
            filters, filters, kernel_size=kernel, padding=0)
        self.bn2 = nn.BatchNorm2d(filters)
        self.negative_pad = nn.ConstantPad2d(1, -1)

    def forward(self, x):
        inp = x

        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(self.negative_pad(x))

        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(self.negative_pad(x))

        x = x + inp

        return x

class convnet(nn.Module):
    def __init__(self, board_size=20, hidden_size=128, conv_channel = 64, in_channel=3, last_conv_channel=32):
        super(convnet, self).__init__()
        self.hidden_size = hidden_size
        self.board_size = board_size
        self.conv_channel = conv_channel
        self.in_channel = in_channel
        self.last_conv_channel = last_conv_channel
        self.negative_pad = nn.ConstantPad2d(1, -1)

        self.conv1 = nn.Conv2d(self.in_channel,self.conv_channel,kernel_size=3,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(self.conv_channel)


        self.conv2 = nn.Conv2d(self.conv_channel,self.conv_channel,kernel_size=3,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(self.conv_channel)

        self.conv3 = nn.Conv2d(self.conv_channel, self.last_conv_channel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.last_conv_channel)

        self.dense = nn.Linear(self.last_conv_channel*self.board_size**2, 4)

    def forward(self,x):
        x = x.view(-1,self.in_channel,self.board_size,self.board_size).float()
        x = F.relu(self.bn1(self.conv1(self.negative_pad(x))))
        x = F.relu(self.bn2(self.conv2(self.negative_pad(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dense(x.view(-1,self.last_conv_channel*self.board_size**2))
        # print('Model output:',x)
        return x


class ResNet(nn.Module):
    def __init__(self, board_size=20, hidden_size=256, conv_channel=64, in_channel=3, last_conv_channel=32, blocks=5):
        super(ResNet, self).__init__()
        self.hidden_size = hidden_size
        self.board_size = board_size
        self.conv_channel = conv_channel
        self.in_channel = in_channel
        self.last_conv_channel = last_conv_channel
        self.blocks = blocks
        self.negative_pad = nn.ConstantPad2d(1, -1)

        self.init_conv = nn.Conv2d(self.in_channel,self.conv_channel,kernel_size=3,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(self.conv_channel)

        self.res_blocks = nn.ModuleList([ResBlock(filters=conv_channel) for i in range(blocks-1)])

        self.last_conv = nn.Conv2d(self.conv_channel, self.last_conv_channel, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(self.last_conv_channel)

        self.hidden = nn.Linear(self.last_conv_channel*self.board_size**2, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 4)

    def forward(self,x):
        x = x.view(-1,self.in_channel,self.board_size,self.board_size).float()
        x = F.relu(self.bn1(self.init_conv(self.negative_pad(x))))
        for i in range(self.blocks-1):
            x = self.res_blocks[i](x)
        x = F.relu(self.bn2(self.last_conv(x)))
        x = self.hidden(x.view(-1,self.last_conv_channel*self.board_size**2))
        x = self.q(x)
        # print('Model output:',x)
        return x