import argparse

import torch
import numpy as np
import os
import time
from train_model import Trainer
from parsing import add_args

parser = argparse.ArgumentParser()
add_args(parser)

args = parser.parse_args()
#args.use_cuda = torch.cuda.is_available()




def main():
    trainer = Trainer(args=vars(args))
    trainer.ggr(False)

if __name__ == '__main__':
    main()
