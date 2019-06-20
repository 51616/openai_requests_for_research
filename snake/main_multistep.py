from game import Snake
from model import feedforward, convnet
from trainer import ReplayMemory, select_action, optimize_model
from utils import Transition
import config
from config import device

from collections import deque
import numpy as np
import torch
from itertools import count
import time



policy_net = convnet(config.BOARD_SIZE, in_channel=5).float().to(device, non_blocking=True).eval()
target_net = convnet(config.BOARD_SIZE, in_channel=5).float().to(device, non_blocking=True).eval()
optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [25000,50000,75000,100000,125000,200000], gamma=0.5)

env = Snake(config.BOARD_SIZE)

replay_memory = ReplayMemory(capacity=100000)

n_steps = config.START_N_STEPS

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
    

    rewards = deque([], maxlen = n_steps)
    states = deque([], maxlen = n_steps)
    actions = deque([], maxlen = n_steps)

    for step in count(1):

        action = select_action(obs, policy_net, steps_done)
        steps_done += 1

        new_obs, reward, done = env.step(action)
        cum_reward += reward
        # rewards = deque([r/config.GAMMA for r in rewards], maxlen = n_steps)
        # rewards.append( reward * config.GAMMA**n_steps )
        rewards.append(reward)
        states.append(obs)
        actions.append(action)

        discounted_reward = list(rewards)

        if done:
            for i in range(len(states)):
                # n_steps_reward = np.sum(discounted_reward[i:])
                n_steps_reward = 0
                for td,j in enumerate(range(i,len(rewards))):
                    n_steps_reward += (config.GAMMA**td) * rewards[j]
                transition = Transition(torch.tensor(states[i]).to(device, non_blocking=True), torch.tensor([actions[i]]).to(device, non_blocking=True),
                                        None, torch.tensor([n_steps_reward]).to(device, non_blocking=True).float())
                # print(transition)
                replay_memory.push(transition)

        elif len(states)==n_steps:
                # n_steps_reward = np.sum(discounted_reward)
                n_steps_reward = 0
                for i in range(n_steps):
                    n_steps_reward += (config.GAMMA**i) * rewards[i]
                transition = Transition(torch.tensor(states[0]).to(device, non_blocking=True), torch.tensor([actions[0]]).to(device, non_blocking=True),
                                        torch.tensor(new_obs).to(device, non_blocking=True), torch.tensor([n_steps_reward]).to(device, non_blocking=True).float())
                # print(transition)
                replay_memory.push(transition)

            

        if (steps_done%config.STEP_SIZE==0) and (len(replay_memory)>=10000):
            # print('Training...')
            optimize_model(policy_net, target_net, replay_memory, optimizer, scheduler)

        if (steps_done%config.N_STEPS_UPDATE==0) and (steps_done>1):
            n_steps = min(n_steps+1, config.MAX_N_STEPS)
            print('Update n-steps to :',n_steps)

        if done:
            break

        obs = new_obs

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