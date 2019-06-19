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

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def plot_rewards():
    global means
    plt.figure(2)
    plt.clf()
    # durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Episode reward')
    plt.plot(episode_rewards)
    #plt.plot(episode_durations)
    
    # Take 100 episode averages and plot them too
    means.append(np.mean(episode_rewards[-100:]))
    plt.plot(means)
    # if len(episode_durations) >= 100:
    #     # means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     # means = torch.cat((torch.zeros(99), means))
    #     means.append(np.mean(episode_durations[-100:]))
    #     plt.plot(means)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


policy_net = convnet(config.BOARD_SIZE, in_channel=5).float().to(device, non_blocking=True).eval()
target_net = convnet(config.BOARD_SIZE, in_channel=5).float().to(device, non_blocking=True).eval()
optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [25000,50000,75000,100000,125000,175000], gamma=0.5)

env = Snake(config.BOARD_SIZE)

replay_memory = ReplayMemory(capacity=100000)

steps_done = 0
ep = 0
episode_rewards = []
episode_durations = []
means = []

t = time.time()

while (steps_done < config.TOTAL_STEPS):
    done = False
    obs = env.reset()
    cum_reward = 0
    ep += 1
    # render = False
    # if ep % config.SHOW_IT==0:
    #     render = True
    # if render:
    #     env.render()
    for step in count(1):
        # action = np.random.choice(env.action_space)
        # pred = nnet(obs).detach().numpy()
        # action = np.argmax(pred)
        # print('Model mode:',policy_net.training)
        action = select_action(obs, policy_net, steps_done)
        # print(action)
        steps_done += 1

        new_obs, reward, done = env.step(action)
        cum_reward += reward
        
        if not done:
            transition = Transition(torch.tensor(obs).to(device, non_blocking=True), torch.tensor([action]).to(device, non_blocking=True),
                                    torch.tensor(new_obs).to(device, non_blocking=True), torch.tensor([reward]).to(device, non_blocking=True).float())
        else:
            transition = Transition(torch.tensor(obs).to(device, non_blocking=True), torch.tensor([action]).to(device, non_blocking=True),
                                    None, torch.tensor([reward]).to(device, non_blocking=True).float())  # SARSA?
        # print(transition)
        replay_memory.push(transition)

        # print(transition)

        obs = new_obs

        if (steps_done%config.STEP_SIZE==0) and (steps_done>=10000):
            optimize_model(policy_net, target_net, replay_memory, optimizer, scheduler)
        # if (steps_done%config.TARGET_UPDATE==0):
        #     target_net.load_state_dict(policy_net.state_dict())

        # if render:
        #     env.render()
        if done:
            # if render:
            #     cv2.destroyAllWindows()
            break

    episode_rewards.append(cum_reward)
    # Update the target network, copying all weights and biases in DQN
    if ep % config.TARGET_UPDATE == 0:
        print('EPISODES:',ep)
        print('Last 100 episodes mean rewards:',np.mean(episode_rewards[-100:]))
        print('Last 100 episodes max rewards:',np.max(episode_rewards[-100:]))
        print('Last 100 episodes min rewards:',np.min(episode_rewards[-100:]))
        print('Total steps:',steps_done)
        print(steps_done / (time.time()-t),'FPS')
        print()
        target_net.load_state_dict(policy_net.state_dict())
        # t = time.time()
        


torch.save({
            'episodes': ep,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            '100eps_mean_reward': np.mean(episode_rewards[-100:]),
            }, 'policy_net.pth')