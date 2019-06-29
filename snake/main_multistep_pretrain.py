from game import Snake
from model import feedforward, convnet, convnet_duel
from trainer import ReplayMemory, select_action, optimize_model
from utils import Transition
import config
from config import device

from collections import deque
import numpy as np
import torch
from itertools import count
import time



policy_net = convnet(config.BOARD_SIZE, in_channel=5).float().to(device, non_blocking=True)
target_net = convnet(config.BOARD_SIZE, in_channel=5).float().to(device, non_blocking=True)

# print(policy_net)
# print(policy_net.state_dict()['conv1.weight'])

pretrain_checkpoint = torch.load('policy_net_10_step_no_bug.pth', map_location=device)
pretrained_dict = {k: v for k, v in pretrain_checkpoint['model_state_dict'].items() if 'dense' not in k} # load conv layer weights

policy_net.load_state_dict(pretrained_dict, strict=False)
policy_net.eval()
target_net.load_state_dict(pretrained_dict, strict=False)
target_net.eval()


optimizer = torch.optim.RMSprop([{'params':policy_net.conv1.parameters(), 'lr':1e-6},
                                {'params':policy_net.bn1.parameters(), 'lr':1e-6},
                                {'params':policy_net.conv2.parameters(), 'lr':1e-6},
                                {'params':policy_net.bn2.parameters(), 'lr':1e-6},
                                {'params':policy_net.dense.parameters()}] , lr=1e-4, momentum=0.9)

def constant_lr(x):
    return 1

def multi_step_lr(iteration):
    milestones = [25000,50000,75000,100000,125000,250000,500000]
    gamma = 0.5
    for i, milestone in enumerate(milestones[::-1]):
        if iteration >= milestone:
            return gamma**(len(milestones)-i)
    return 1




scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    lr_lambda=[constant_lr,constant_lr,constant_lr,constant_lr,multi_step_lr])

env = Snake(config.BOARD_SIZE)

replay_memory = ReplayMemory(capacity=100000)

n_steps = config.START_N_STEPS

steps_done = 0
ep = 0
episode_rewards = []
episode_durations = []
means = []
best_mean = -1

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
        
        rewards.append( reward * config.GAMMA**n_steps )
        rewards = deque([r/config.GAMMA for r in rewards], maxlen = n_steps)
        
        # rewards.append(reward)
        states.append(obs)
        actions.append(action)

        discounted_reward = list(rewards)

        # x= torch.tensor(np.arange(8)).view(2,2,2)
        # x90 = x.transpose(0, 1).flip(0)
        # x180 = x.flip(0).flip(1)
        # x270 = x.transpose(0, 1).flip(1)

        if done:
            for i in range(len(states)):
                n_steps_reward = np.sum(discounted_reward[i:])
                # n_steps_reward = 0
                # for td,j in enumerate(range(i,len(rewards))):
                #     n_steps_reward += (config.GAMMA**td) * rewards[j]
                transition = Transition(torch.tensor(states[i]).to(device, non_blocking=True), torch.tensor([actions[i]]).to(device, non_blocking=True),
                                        None, torch.tensor([n_steps_reward]).to(device, non_blocking=True).float())
                # print(transition)
                replay_memory.push(transition)

        elif len(states)==n_steps:
                n_steps_reward = np.sum(discounted_reward)
                # n_steps_reward = 0
                # for i in range(n_steps):
                #     n_steps_reward += (config.GAMMA**i) * rewards[i]
                transition = Transition(torch.tensor(states[0]).to(device, non_blocking=True), torch.tensor([actions[0]]).to(device, non_blocking=True),
                                        torch.tensor(new_obs).to(device, non_blocking=True), torch.tensor([n_steps_reward]).to(device, non_blocking=True).float())
                # print(transition)
                replay_memory.push(transition)

        obs = new_obs

        if (steps_done%config.STEP_SIZE==0) and (len(replay_memory)>=10000):
            # print('Training...')
            optimize_model(policy_net, target_net, replay_memory, optimizer, scheduler, n_steps)

        if (steps_done%config.N_STEPS_UPDATE==0) and (steps_done>1):
            n_steps = min(n_steps+1, config.MAX_N_STEPS)
            print('Update n-steps to :',n_steps)

        if done:
            break

        

    episode_rewards.append(cum_reward)
    # Update the target network, copying all weights and biases in DQN
    if ep % config.TARGET_UPDATE == 0:
        ep_mean = np.mean(episode_rewards[-100:])
        print('EPISODES:',ep)
        print('Last 100 episodes mean rewards:',ep_mean)
        print('Last 100 episodes max rewards:',np.max(episode_rewards[-100:]))
        print('Last 100 episodes min rewards:',np.min(episode_rewards[-100:]))
        print('Total steps:',steps_done)
        print(steps_done / (time.time()-t),'FPS')
        print()
        target_net.load_state_dict(policy_net.state_dict())
        if (ep_mean>best_mean):
            torch.save({
            'episodes': ep,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            '100eps_mean_reward': np.mean(episode_rewards[-100:]),
            }, 'best_dqn.pth')
        # t = time.time()
        


torch.save({
            'episodes': ep,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            '100eps_mean_reward': np.mean(episode_rewards[-100:]),
            }, 'lastest_dqn.pth')