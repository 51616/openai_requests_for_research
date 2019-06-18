import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

BOARD_SIZE = 7

NUM_STEPS = 1000000
STEP_SIZE = 4
BATCH_SIZE = 128 * STEP_SIZE
GAMMA = 0.999999
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 10000
TARGET_UPDATE = 100
SHOW_IT = 10000