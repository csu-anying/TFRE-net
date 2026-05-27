import torch
print(torch.__version__)
# from torch import true_divide
# from pre_train import train
from checked_trans import train # NB-net
# from checked_transNew import train
# from check_wrong import train # SPG-net
import sys

if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('e.g., CUDA_VISIBLE_DEVICES=? python main.py PTB/PTBXL 0/1')

    if int(sys.argv[2]): test = False
    else: test = True
    # torch.manual_seed(0)
    print('Running training code...')
    train(test, 7, sys.argv[1])
    print('Done.')