import json
import os
import numpy as np
import networkx as nx
from metrics import get_statistics
import random
import torch
import argparse
import time
from model_utils import *
from cana_model import *
import numba

def parse_args():
    parser = argparse.ArgumentParser(description="CANA UIL")
    parser.add_argument('--emb_dim',    default=150,       type=int)
    parser.add_argument('--layers',     default=2,         type=int)
    parser.add_argument('--fea_epochs', default=20,        type=int)
    parser.add_argument('--fea_batch_size', default=64,        type=int)
    parser.add_argument('--fea_lr', default=1e-5,        type=float)
    parser.add_argument('--nei_epochs', default=30,        type=int)
    parser.add_argument('--nei_lr', default=5e-3,        type=float)
    parser.add_argument('--ppmi_step', default=10,        type=int)
    parser.add_argument('--ppmi_trans_ratio', default=0.98,        type=float)
    parser.add_argument('--norm_factor', default=5,        type=int)
    parser.add_argument('--dataset',    default='douban',  type=str)
    parser.add_argument('--seed',       default=666,        type=int)
    return parser.parse_args()



if __name__=='__main__':
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    src_g,src_f,trg_g,trg_f,training_set,test_set,gth = load_data(args)
    print('LOAD DATA OK!!!')
    print(src_g.number_of_edges(),trg_g.number_of_edges())

    S = CANA_algorithm(src_g,src_f,trg_g,trg_f,training_set, args, test_set,gth)
    
    a1,p1,p5 = get_statistics(S,test_set,gth)
    print('ACC={:.4f}   Hits@1={:.4f}   Hits@5={:.4f}'.format(a1,p1,p5))

    