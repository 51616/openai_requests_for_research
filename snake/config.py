import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

BOARD_SIZE = 7

NUM_STEPS = 10000000
STEP_SIZE = 4
BATCH_SIZE = 512
GAMMA = 0.999999
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 10000
TARGET_UPDATE = 1000
SHOW_IT = 10000