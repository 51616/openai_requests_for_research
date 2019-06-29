from game import Snake
from model import feedforward, convnet
from trainer import ReplayMemory, select_action, optimize_model
from utils import Transition
import config
from config import device

from collections import namedtuple
import numpy as np
import random
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
from itertools import count
import time


checkpoint = torch.load('policy_net_10_step_no_bug.pth', map_location=device)

policy_net = convnet(config.BOARD_SIZE, in_channel=5).float().to(device, non_blocking=True)
policy_net.load_state_dict(checkpoint['model_state_dict'])
policy_net.eval()

env = Snake(config.BOARD_SIZE)


while 1:
    done = False
    obs = env.reset()
    cum_reward = 0
    render = True
    env.render()
    for step in count(1):
        action = select_action(obs, policy_net, 0, explore=False)
        new_obs, reward, done = env.step(action)
        cum_reward += reward
        obs = new_obs

        env.render()

        if done:
            cv2.destroyAllWindows()
            break
    print('Reward:',cum_reward)