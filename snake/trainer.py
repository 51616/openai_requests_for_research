import random
import torch
import torch.nn.functional as F
import config
from config import device
import math
import numpy as np
from utils import Transition

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, obj):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = obj
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(obs, policy_net, steps_done, explore=True):
    if explore:
        sample = random.random()
        eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(-1. * steps_done / config.EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.tensor(obs).to(device, non_blocking=True)
                return np.argmax(policy_net(state).detach().cpu().numpy())
        else:
            return np.random.randint(4)  # torch.tensor([[random.randrange(4)]])  # , device=device, dtype=torch.long
    else:
        state = torch.tensor(obs).to(device, non_blocking=True)
        pred = policy_net(state)
        # print(pred)
        return np.argmax(pred.detach().cpu().numpy())


def optimize_model(policy_net, target_net, replay_memory, optimizer, scheduler):
    if len(replay_memory) < config.BATCH_SIZE:
        return
    # print('Training...')
    policy_net.train()
    # print('Model mode:',policy_net.training)
    transitions = replay_memory.sample(config.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8) 
    
    try:
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    except:
        non_final_next_states = None

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    online_prediction = policy_net(state_batch)
    state_action_values = online_prediction.gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros((config.BATCH_SIZE,1), device=device)
    
    if non_final_next_states is not None :
        #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].view(-1,1).detach()

        # Double DQN
	    next_state_action = online_prediction[non_final_mask].max(1)[1].view(-1,1)
	    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_action).float()
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch.float()
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    # for param_group in optimizer.param_groups:
    #     print(param_group['lr'])
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    scheduler.step()
    policy_net.eval()
    # print('Model mode:',policy_net.training)
    return