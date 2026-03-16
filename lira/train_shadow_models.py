import os
import sys
import torch
import numpy as np
sys.path.append(os.getcwd())
from core.args import LiRA_Shadow_Parser
from core.lira import train_shadow

if __name__ == "__main__":
    parser = LiRA_Shadow_Parser()
    args = parser.parse_args()
    for i in range(args.num_shadow_models):
        train_shadow(args, i)
