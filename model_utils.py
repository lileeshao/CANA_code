import json
import os
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import random



def load_graph(graph_path,feature_path):
    G_data = json.load(open(graph_path))
    G = json_graph.node_link_graph(G_data)
    feat = np.load(feature_path, allow_pickle=True)

    return G,feat

def load_train(base_dir):
    with open(os.path.join(base_dir, 'train')) as f:
        train_pairs = f.readlines()

    random.shuffle(train_pairs)
    train_data = np.zeros((len(train_pairs), 2), dtype=int)

    for i in range(len(train_pairs)):
        s,t = train_pairs[i].strip().split()
        train_data[i][0] = int(s)
        train_data[i][1] = int(t)

    return train_data

def load_groundtruth(base_dir, snode, tnode):
    groundtruth = np.zeros((snode, tnode))

    with open(os.path.join(base_dir, 'test')) as f:
        for line in f:
            s,t = line.strip().split()
            groundtruth[int(s)][int(t)] = 1

    return groundtruth

def load_groundtruth_dict(base_dir, snode, tnode):
    groundtruth = {}

    with open(os.path.join(base_dir, 'test')) as f:
        for line in f:
            s,t = line.strip().split()
            groundtruth[int(s)] = int(t)

    return groundtruth

def load_data(args):
    base_dir = './data/'+args.dataset
    src_g,src_f = load_graph(os.path.join(base_dir,'source.json'), os.path.join(base_dir,'source_feats.npy'))
    trg_g,trg_f = load_graph(os.path.join(base_dir,'target.json'), os.path.join(base_dir,'target_feats.npy'))
    train_data = load_train(base_dir)
    test_data = load_groundtruth(base_dir,src_g.number_of_nodes(), trg_g.number_of_nodes())
    gth = load_groundtruth_dict(base_dir,src_g.number_of_nodes(), trg_g.number_of_nodes())
    
    print(train_data.shape)
    return src_g,src_f,trg_g,trg_f,train_data,test_data,gth


