import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

BOARD_SIZE = 7

TOTAL_STEPS = 2000000
STEP_SIZE = 4
BATCH_SIZE = 512
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 10000
TARGET_UPDATE = 100
SHOW_IT = 10000
START_N_STEPS = 1
MAX_N_STEPS = 10
N_STEPS_UPDATE = 100000