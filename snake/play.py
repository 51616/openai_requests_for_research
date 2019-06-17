from game import Snake
from model import feedforward, convnet
from trainer import ReplayMemory, select_action, optimize_model
from utils import Transition
import config

from collections import namedtuple
import numpy as np
import random
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
from itertools import count
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('policy_net.pth')

policy_net = convnet(config.BOARD_SIZE).float().to(device, non_blocking=True)
policy_net.load_state_dict(checkpoint['model_state_dict'])
policy_net.eval()

env = Snake(config.BOARD_SIZE)


while 1:
    done = False
    obs = env.reset()
    cum_reward = 0
    render = True
    if render:
        env.render()
    for step in count(1):
        action = select_action(obs, policy_net, 0, explore=False)
        new_obs, reward, done = env.step(action)
        cum_reward += reward
        obs = new_obs

        if render:
            env.render()
        if done:
            if render:
                cv2.destroyAllWindows()
            break
    print('Reward:',cum_reward)