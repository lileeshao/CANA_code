import os
import numpy as np
import networkx as nx
import random
import torch_geometric as pyg
import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
import tqdm

from model_utils import *
from metrics import get_statistics
import sklearn.metrics.pairwise as pw
import time
import pdb
from scipy.sparse import csr_matrix


def pairwise_dist(x):
    eps=1e-9
    instances_norm = torch.sum(x**2, -1).reshape((-1,1))

    output = -2*torch.mm(x, x.t()) 
    output += instances_norm 
    output += instances_norm.t()
    return torch.sqrt(output.clamp(min=0) + eps)

def dcor(x, y):
    eps=1e-9
    m,_ = x.shape

    assert len(x.shape) == 2
    assert len(y.shape) == 2

    dx = pairwise_dist(x)
    dy = pairwise_dist(y)

    dx_m = dx - dx.mean(dim=0)[None, :] - dx.mean(dim=1)[:, None] + dx.mean()
    dy_m = dy - dy.mean(dim=0)[None, :] - dy.mean(dim=1)[:, None] + dy.mean()

    dcov2_xy = (dx_m * dy_m).sum()/float(m * m) 
    dcov2_xx = (dx_m * dx_m).sum()/float(m * m) 
    dcov2_yy = (dy_m * dy_m).sum()/float(m * m) 

    dcor = torch.sqrt(dcov2_xy)/torch.sqrt((torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy)).clamp(min=0) + eps)

    return eps+dcor/float(m*m)





class CANA_fea(nn.Module):
    def __init__(self, in_dim):
        super(CANA_fea, self).__init__()
        self.p = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh()
            )

        self.q = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh()
            )

    def forward(self, src_fea, trg_fea):
        src_p_fea = self.p(src_fea)
        src_q_fea = self.q(src_fea)

        trg_p_fea = self.p(trg_fea)
        trg_q_fea = self.q(trg_fea)

        return src_p_fea,src_q_fea,trg_p_fea,trg_q_fea

    def loss(self,src_p_fea,src_q_fea,trg_p_fea,trg_q_fea,node_pairs):
        pair_num = node_pairs.shape[0]
        f1 = F.normalize(src_p_fea[node_pairs[:,0]])
        f2 = F.normalize(trg_p_fea[node_pairs[:,1]])
        neg_f2 = f2[torch.randperm(pair_num)]

        loss_anchor=torch.sum(torch.mul(f1,f2),1)-torch.sum(torch.mul(f1,neg_f2),1)
        loss_anchor = -torch.mean(loss_anchor)

        p_p1 = torch.mean(src_q_fea,0)
        p_p2 = torch.mean(trg_q_fea,0)
        loss_center=-torch.norm(p_p1-p_p2)
        
        loss_dcor = dcor(src_p_fea,src_q_fea)+dcor(trg_p_fea,trg_q_fea)

        return loss_anchor+loss_center+loss_dcor




class CANA_nei(nn.Module):
    def __init__(self, in_dim, args, device, src_ppmi, trg_ppmi):
        super(CANA_nei, self).__init__()
        self.hidden_dim = args.emb_dim
        self.layers = args.layers
        self.device = device
        hid1 = args.emb_dim

        self.src_ppmi = src_ppmi
        self.trg_ppmi = trg_ppmi

        self.fx = nn.Sequential(nn.Linear(in_dim,hid1),
            nn.ReLU(),
            nn.Linear(hid1,hid1),
            nn.ReLU())
        
        #self.gx = nn.Sequential(nn.Linear(in_dim,hid1),
        #    nn.ReLU(),
        #    nn.Linear(hid1,hid1),
        #    nn.ReLU())

        self.convs = nn.ModuleList()
        for i in range(self.layers):
            in_dims = hid1 if i == 0 else self.hidden_dim
            self.convs.append(GraphConv(in_dims, self.hidden_dim,bias=False))
        
        self.edge_att_mlp = nn.Sequential(nn.Linear(in_dims*2,20),
            nn.ReLU(),
            nn.Linear(20,1),
            nn.ReLU())
        
        self.act = nn.Softsign()

    def forward(self, src_g, src_fea, trg_g, trg_fea, hard_src=0, hard_trg=0):
        src_fea1 = self.fx(src_fea)
        trg_fea1 = self.fx(trg_fea)

        embs_list = [src_fea1]
        embt_list = [trg_fea1]

        confd_src = [None]
        confd_trg = [None]

        src_row,src_col = src_g
        trg_row,trg_col = trg_g


        x = src_fea1
        for i, conv in enumerate(self.convs):
            edge_rep = torch.cat([x[src_row], x[src_col]], dim=-1)
            edge_att = torch.sigmoid(self.edge_att_mlp(edge_rep))

            edge_weight_c = (edge_att+hard_src)/2
            edge_weight_o = 1-edge_weight_c

            x = self.act(conv(x, src_g, edge_weight_c))
            cfd = self.act(conv(x, src_g, edge_weight_o))

            embs_list.append(x)
            confd_src.append(cfd)

        x = trg_fea1
        for i, conv in enumerate(self.convs):
            edge_rep = torch.cat([x[trg_row], x[trg_col]], dim=-1)
            edge_att = torch.sigmoid(self.edge_att_mlp(edge_rep))

            edge_weight_c = (edge_att+hard_trg)/2
            edge_weight_o = 1-edge_weight_c

            x = self.act(conv(x, trg_g, edge_weight_c))
            cfd = self.act(conv(x, trg_g, edge_weight_o))

            embt_list.append(x)
            confd_trg.append(cfd)
        
        return embs_list,embt_list,confd_src,confd_trg

    def ppmi_loss(self, embedding, ppmi):
        ppmi = ppmi-1
        adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())

        temp = adj-1 
        adj = F.normalize(temp, dim = 1)

        ppmi_loss = (adj - ppmi) ** 2
        ppmi_loss = ppmi_loss.sum() / ppmi.shape[1]
        return ppmi_loss

    def emb_loss(self,src_emb,trg_emb):
        l = 0
        for i, (es, et) in enumerate(zip(src_emb,trg_emb)):               
            loss_s = self.ppmi_loss(es, self.src_ppmi)
            loss_t = self.ppmi_loss(et, self.trg_ppmi)
            l += loss_s + loss_t

        return l

    def bpr_loss(self,src_emb,trg_emb,cfd_src,cfd_trg,src_pos,trg_pos,src_neg,trg_neg):
        l = 0
        anum = src_pos.shape[0]
        for i, (es, et, cfs, cft) in enumerate(zip(src_emb,trg_emb,cfd_src,cfd_trg)):
            if i == 0 :
                continue         
            scpemb,tcpemb = es[src_pos],et[trg_pos]       
            sfpemb,tfpemb = cfs[src_pos[torch.randperm(anum)]],cft[trg_pos[torch.randperm(anum)]]   

            spemb = scpemb+sfpemb
            tpemb = tcpemb+tfpemb
            pos_sim = torch.sum(torch.mul(spemb,tpemb), dim=1)

            scnemb,tcnemb = es[src_neg],et[trg_neg]       
            sfnemb,tfnemb = cfs[src_neg[torch.randperm(anum)]],cft[trg_neg[torch.randperm(anum)]]   

            snemb = scnemb+sfnemb
            tnemb = tcnemb+tfnemb
            neg_sim = torch.sum(torch.mul(snemb,tnemb), dim=1)  

            l += torch.mean(-torch.log(torch.sigmoid(pos_sim-neg_sim)))

        return l



def scale_matrix(mat):
    mat = mat - np.diag(np.diag(mat))
    #D_inv = np.diagflat(np.reciprocal(np.sum(mat, axis=0)))
    D_inv = 1/(np.sum(mat, axis=0)+1e-9)
    #print(np.max(D_inv), np.sum(D_inv))
    mat = np.multiply(D_inv,mat)

    return mat



def random_surfing(adj_matrix, step, ratio):
    # Random Surfing
    num_node = adj_matrix.shape[0]

    adj_matrix = scale_matrix(adj_matrix)
    P0 = np.eye(num_node, dtype="float32")
    M = np.zeros((num_node, num_node), dtype="float32")
    P = np.eye(num_node, dtype="float32")
    for i in range(0, step):
        P = ratio * np.dot(P, adj_matrix) + (1 - ratio) * P0
        M = M + P
        
    return M

def random_surfing_debug(adj_matrix, step, ratio):
    # for debug, not use
    num_node = adj_matrix.shape[0]

    #adj_matrix = scale_matrix(adj_matrix)
    P0 = np.eye(num_node, dtype="float32")
    M = np.zeros((num_node, num_node), dtype="float32")
    P = np.ones(num_node, dtype="float32")/num_node
    for i in range(0, step):
        P = ratio * np.dot(P, adj_matrix) + (1 - ratio) * P0
        M = M + P
        
    return M



def get_ppmi_matrix(mat, step, ratio):
    # Get Positive Pairwise Mutual Information(PPMI) matrix
    num_node = mat.shape[0]
    mat = random_surfing(mat,  step, ratio)
    M = scale_matrix(mat)
    col_s = np.sum(M, axis=0).reshape(1, num_node)
    row_s = np.sum(M, axis=1).reshape(num_node, 1)
    D = np.sum(col_s)
    rowcol_s = np.dot(row_s, col_s)+1e-9
    PPMI = np.log(np.divide(D * M, rowcol_s)+1e-9)

    PPMI[np.isnan(PPMI)] = 0.0
    PPMI[np.isinf(PPMI)] = 0.0
    PPMI[np.isneginf(PPMI)] = 0.0
    PPMI[PPMI < 0] = 0.0
    return PPMI



def ppmi_adj(adj,device, args):
    A = adj.todense()
    num_node = A.shape[0]
    I = np.eye(num_node, dtype="float32")  
    A_ppmi = get_ppmi_matrix(A, args.ppmi_step, args.ppmi_trans_ratio)+I
    A_ppmi = torch.FloatTensor(A_ppmi).to(device)

    return A_ppmi




def gen_neg_pairs(src_node_num,trg_node_num,train_pairs,device):
    train_num = train_pairs.shape[0]
    src_l = [i for i in range(src_node_num)]
    random.shuffle(src_l)
    src_neg = np.asarray(src_l[:train_num])

    trg_l = [j for j in range(trg_node_num)]
    random.shuffle(trg_l)
    trg_neg = np.asarray(trg_l[:train_num])

    tr_src = train_pairs[:,0]
    tr_trg = train_pairs[:,1]

    src_neg = torch.tensor(np.concatenate((tr_src,src_neg),axis=0)).long().to(device)
    trg_neg = torch.tensor(np.concatenate((trg_neg,tr_trg),axis=0)).long().to(device)

    return src_neg,trg_neg



def nei_warmup(src_g, trg_g, src_fea, trg_fea, train_data, src_pos,trg_pos, args, device, src_node_num, trg_node_num, source_graph,target_graph, train_pairs):
    """device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    src_node_num = source_graph.number_of_nodes()
    trg_node_num = target_graph.number_of_nodes()
    print('node:',src_node_num,trg_node_num)

    src_g = pyg.utils.from_networkx(source_graph).edge_index.to(device)
    trg_g = pyg.utils.from_networkx(target_graph).edge_index.to(device)
    src_fea = torch.Tensor(source_feature).to(device)
    trg_fea = torch.Tensor(target_feature).to(device)

    train_data = np.concatenate((train_pairs,train_pairs),axis=0)
    src_pos = torch.tensor(train_data[:,0]).long().to(device)
    trg_pos = torch.tensor(train_data[:,1]).long().to(device)"""

    #train_data = torch.from_numpy(train_pairs).to(device)
    #train_data = train_data.long()
    #train_labels = torch.from_numpy(np.ones((train_pairs.shape[0],))).to(device)
    #train_labels = train_labels.float()
    print('GET PPMI')

    src_ppmi = ppmi_adj(nx.adjacency_matrix(source_graph), device, args)
    trg_ppmi = ppmi_adj(nx.adjacency_matrix(target_graph), device, args)
    print("PPMI OK")

    emb_model = CANA_nei(src_fea.shape[-1],args,device, src_ppmi, trg_ppmi)
    emb_model = emb_model.to(device)

    optimizer = torch.optim.Adam(emb_model.parameters(), lr=args.nei_lr)

    #warmup
    for ep in range(args.nei_epochs):
        src_neg,trg_neg = gen_neg_pairs(src_node_num,trg_node_num,train_pairs,device)
        optimizer.zero_grad()
        src_emb,trg_emb,cfd_src,cfd_trg = emb_model.forward(src_g, src_fea, trg_g, trg_fea)
        emb_l = emb_model.emb_loss(src_emb,trg_emb)
        anchor_l = emb_model.bpr_loss(src_emb,trg_emb,cfd_src,cfd_trg,src_pos,trg_pos,src_neg,trg_neg)
        l = emb_l + anchor_l
        l.backward()
        optimizer.step()

    """
    for ep in range(args.nei_epochs):
        src_neg,trg_neg = gen_neg_pairs(src_node_num,trg_node_num,train_pairs,device)
        optimizer.zero_grad()
        src_emb,trg_emb,cfd_src,cfd_trg = emb_model.forward(src_g, src_fea, trg_g, trg_fea)
        l = emb_model.emb_loss(src_emb,trg_emb)+emb_model.bpr_loss(src_emb,trg_emb,cfd_src,cfd_trg,src_pos,trg_pos,src_neg,trg_neg)
        optimizer.step()
    """

    return emb_model,optimizer



def sim_calculation(soft_sim,src_g,trg_g,mapping_vec,src_unaligned,trg_unaligned,norm_factor):
    src_list = list(src_unaligned)
    trg_list = list(trg_unaligned)
    sim = np.zeros((len(src_list)*len(trg_list),3))
    nnd = src_g.number_of_nodes()+trg_g.number_of_nodes()

    mapping_vec_r = [-(i+1) for i in range(trg_g.number_of_nodes())]
    mapping_vec_r = np.asarray(mapping_vec_r)
    for i in range(mapping_vec.shape[0]):
        if mapping_vec[i]>=0:
            mapping_vec_r[mapping_vec[i]] = i

    idnum = 0
    for snode in src_list:
        src_neighbor = [mapping_vec[nod] for nod in src_g.neighbors(snode)]
        #v1 = csr_matrix((np.ones(len(src_neighbor),), (np.zeros(len(src_neighbor),), src_neighbor)), shape=(1,nnd))
        v1 = np.zeros(nnd)
        v1[src_neighbor] = 1.0
        #v1 = csr_matrix(v1)
        for tnode in trg_list:
            sim[idnum][0] = int(snode)
            sim[idnum][1] = int(tnode)
            trg_neighbor = [nod for nod in trg_g.neighbors(tnode)]
            if mapping_vec_r[tnode]>=0:
                ali_neighbor = [mapping_vec[nod] for nod in src_g.neighbors(mapping_vec_r[tnode])]
                trg_neighbor  += ali_neighbor
            if len(set(src_neighbor)&set(trg_neighbor))<1:
                hard_sim = 0.1
            else:
                v2 = np.zeros(nnd)
                v2[trg_neighbor] = 1.0
                #v2 = csr_matrix(v2)
                hard_sim = 0.1+(np.dot(v1,v2)/(np.sum(v1/norm_factor+v2)+0.01))

            sim[idnum][2] = soft_sim[int(snode)][int(tnode)]*hard_sim
            idnum +=1
    
    return sim



def cal_sim_matrix(soft_sim,src_g,trg_g,mapping_vec,norm_factor,eps=0.1):
    hard_sim = np.zeros((src_g.number_of_nodes(),trg_g.number_of_nodes()))+eps
    nnd = src_g.number_of_nodes()+trg_g.number_of_nodes()
    trg2src = {}
    for i in range(mapping_vec.shape[0]):
        if mapping_vec[i]>=0:
            trg2src[mapping_vec[i]] = i

    src_list = list(src_g.nodes())
    trg_list = list(trg_g.nodes())
    tp=0
    for snode in src_list:
        src_neighbor = [mapping_vec[nod] for nod in src_g.neighbors(snode)]
        v1 = np.zeros(nnd)
        v1[src_neighbor] = 1.0
        #v1 = csr_matrix(v1)
        #tp+=1
        #if tp%500==1:
        #    print(tp)
        for tnode in trg_list:
            trg_neighbor = [nod for nod in trg_g.neighbors(tnode)]
            if tnode in trg2src.keys():
                ali_neighbor = [mapping_vec[nod] for nod in src_g.neighbors(trg2src[tnode])]
                trg_neighbor  += ali_neighbor
            if len(set(src_neighbor)&set(trg_neighbor))>0:
                v2 = np.zeros(nnd)
                v2[trg_neighbor] = 1.0
                #v2 = csr_matrix(v2)
                hard_sim_score = 0.1+(np.dot(v1,v2)/(np.sum(v1/norm_factor+v2)+0.01))
                hard_sim[int(snode)][int(tnode)] = hard_sim_score

    return soft_sim*hard_sim



def update_hard_attn(hard_src, hard_trg, emb_sim,source_graph,target_graph,mapping_vec,choosing_num,src_unaligned,trg_unaligned,norm_factor):
    Sim = sim_calculation(emb_sim,source_graph,target_graph,mapping_vec,src_unaligned,trg_unaligned,norm_factor)
    src_cand_set = set()
    trg_cand_set = set()

    sim = Sim[np.argsort(-Sim[:, 2])]
    map_vec = mapping_vec

    for i in range(sim.shape[0]):
        snode = sim[i][0]
        tnode = sim[i][1]
        if snode not in src_cand_set and tnode not in trg_cand_set:
            map_vec[int(snode)] = int(tnode)
            src_cand_set.add(snode)
            trg_cand_set.add(tnode)
            if len(src_cand_set)>=choosing_num:
                break

    src_idx = pyg.utils.from_networkx(source_graph).edge_index
    for i in range(src_idx.shape[1]):
        if int(src_idx[0][i]) in src_cand_set or int(src_idx[1][i]) in src_cand_set:
            hard_src[i] = 1.0
    trg_idx = pyg.utils.from_networkx(target_graph).edge_index
    for i in range(trg_idx.shape[1]):
        if int(trg_idx[0][i]) in trg_cand_set or int(trg_idx[1][i]) in trg_cand_set:
            hard_trg[i] = 1.0
    #print(torch.sum(hard_src), torch.sum(hard_trg))

    return map_vec,src_cand_set,trg_cand_set, hard_src, hard_trg

def CANA_algorithm(source_graph,source_feature,target_graph, target_feature, train_pairs, args, test_set,gth):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb_model = CANA_fea(source_feature.shape[-1])
    emb_model = emb_model.to(device)

    src_fea = torch.Tensor(source_feature).to(device)
    trg_fea = torch.Tensor(target_feature).to(device)
    train_data = torch.from_numpy(train_pairs).to(device)
    train_data = train_data.long()
    optimizer = torch.optim.Adam(emb_model.parameters(), lr=args.fea_lr)

    for ep in range(args.fea_epochs):
        total = 0
        while total<train_pairs.shape[0]:
            node_pairs = train_data[total:total+args.fea_batch_size]
            src_p_fea,src_q_fea,trg_p_fea,trg_q_fea = emb_model.forward(src_fea,trg_fea)

            optimizer.zero_grad()
            l = emb_model.loss(src_p_fea,src_q_fea,trg_p_fea,trg_q_fea, node_pairs)
            l.backward()
            optimizer.step()
            total+=args.fea_batch_size

    f1,_,f2,_ = emb_model.forward(src_fea,trg_fea)
    source_feature,target_feature = f1.cpu().detach().numpy(),f2.cpu().detach().numpy()

    training_set = train_pairs
    src_node_num = source_graph.number_of_nodes()
    trg_node_num = target_graph.number_of_nodes()
    #('node:',src_node_num,trg_node_num)

    src_g = pyg.utils.from_networkx(source_graph).edge_index.to(device)
    trg_g = pyg.utils.from_networkx(target_graph).edge_index.to(device)
    src_fea = torch.Tensor(source_feature).to(device)
    trg_fea = torch.Tensor(target_feature).to(device)
    hard_src = torch.zeros(src_g.shape[1],1).float().to(device)
    hard_trg = torch.zeros(trg_g.shape[1],1).float().to(device)

    src_train = set()
    trg_train = set()
    mapping_vec = [-(i+1) for i in range(source_graph.number_of_nodes())]
    mapping_vec = np.asarray(mapping_vec)
    
    for i in range(training_set.shape[0]):
        mapping_vec[training_set[i][0]] = training_set[i][1]
        src_train.add(training_set[i][0])
        trg_train.add(training_set[i][1])

    src_idx = pyg.utils.from_networkx(source_graph).edge_index
    for i in range(src_g.shape[1]):
        if int(src_idx[0][i]) in src_train or int(src_idx[1][i]) in src_train:
            hard_src[i] = 1.0
    trg_idx = pyg.utils.from_networkx(target_graph).edge_index
    for i in range(trg_g.shape[1]):
        if int(trg_idx[0][i]) in trg_train or int(trg_idx[1][i]) in trg_train:
            hard_trg[i] = 1.0
        
    train_data = np.concatenate((train_pairs,train_pairs),axis=0)
    src_pos = torch.tensor(train_data[:,0]).long().to(device)
    trg_pos = torch.tensor(train_data[:,1]).long().to(device)
    emb_model,optimizer = nei_warmup(src_g, trg_g, src_fea, trg_fea, train_data,src_pos,trg_pos, args, device, src_node_num, trg_node_num, source_graph,target_graph, train_pairs)

    e1,e2,_,_ = emb_model.forward(src_g, src_fea, trg_g, trg_fea)
    e1 = torch.stack([F.normalize(e) for e in e1],dim=0)
    e2 = torch.stack([F.normalize(e) for e in e2],dim=0)
    emb_sim = torch.mean(torch.bmm(e1,e2.transpose(2,1)), dim=0).cpu().detach().numpy()
    
    
    src_unaligned = set(source_graph.nodes())-src_train
    trg_unaligned = set(target_graph.nodes())-trg_train
    
    aligned = training_set.shape[0]
    TOTAL_NUM = min(source_graph.number_of_nodes(),target_graph.number_of_nodes())
    cnum = (TOTAL_NUM-aligned)//args.nei_epochs
    epoch = 0
    while epoch<args.nei_epochs:
        src_neg,trg_neg = gen_neg_pairs(src_node_num,trg_node_num,train_pairs,device)
        optimizer.zero_grad()

        mapping_vec,src_cand_set,trg_cand_set, hard_src, hard_trg = update_hard_attn(hard_src, hard_trg, emb_sim,source_graph,target_graph,mapping_vec,cnum,src_unaligned,trg_unaligned, args.norm_factor)
        src_unaligned = src_unaligned-src_cand_set
        trg_unaligned = trg_unaligned-trg_cand_set


        src_emb,trg_emb,cfd_src,cfd_trg = emb_model.forward(src_g, src_fea, trg_g, trg_fea, hard_src, hard_trg)
        emb_l = emb_model.emb_loss(src_emb,trg_emb)
        anchor_l = emb_model.bpr_loss(src_emb,trg_emb,cfd_src,cfd_trg,src_pos,trg_pos,src_neg,trg_neg)
        l = emb_l + anchor_l
        l.backward()
        optimizer.step()

        e1,e2,_,_ = emb_model.forward(src_g, src_fea, trg_g, trg_fea)
        e1 = torch.stack([F.normalize(e) for e in e1],dim=0)
        e2 = torch.stack([F.normalize(e) for e in e2],dim=0)
        emb_sim = torch.mean(torch.bmm(e1,e2.transpose(2,1)), dim=0).cpu().detach().numpy()
        epoch+=1

    print('TRAINING OK')
    """
    while epoch<args.nei_epochs:
        e1,e2,_,_ = emb_model.forward(src_g, src_fea, trg_g, trg_fea)
        e1 = torch.stack([F.normalize(e) for e in e1],dim=0)
        e2 = torch.stack([F.normalize(e) for e in e2],dim=0)
        emb_sim = torch.mean(torch.bmm(e1,e2.transpose(2,1)), dim=0).cpu().detach().numpy()
        src_neg,trg_neg = gen_neg_pairs(src_node_num,trg_node_num,train_pairs,device)
        optimizer.zero_grad()

        Sim = sim_calculation(emb_sim,source_graph,target_graph,mapping_vec,src_unaligned,trg_unaligned)
        mapping_vec,src_cand_set,trg_cand_set = update(Sim,mapping_vec,cnum)
        src_unaligned = src_unaligned-src_cand_set
        trg_unaligned = trg_unaligned-trg_cand_set

        src_emb,trg_emb,cfd_src,cfd_trg = emb_model.forward(src_g, src_fea, trg_g, trg_fea, hard_src, hard_trg)
        emb_l = emb_model.emb_loss(src_emb,trg_emb)
        anchor_l = emb_model.bpr_loss(src_emb,trg_emb,cfd_src,cfd_trg,src_pos,trg_pos,src_neg,trg_neg)
        l = emb_l + anchor_l
        l.backward()
        optimizer.step()

        epoch+=1
    """
    return cal_sim_matrix(emb_sim,source_graph,target_graph,mapping_vec, args.norm_factor)



def node_dis_debug(src_f,trg_f,train_pairs,args):
    """
    only for debug, do not use this code
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb_model = CANA_fea(src_f.shape[-1], src_f.shape[0], trg_f.shape[0], device)
    emb_model = emb_model.to(device)

    src_fea = torch.Tensor(src_f).to(device)
    trg_fea = torch.Tensor(trg_f).to(device)
    train_data = torch.from_numpy(train_pairs).to(device)
    train_data = train_data.long()
    optimizer = torch.optim.Adam(emb_model.parameters(), lr=args.fea_lr)

    for ep in range(args.fea_epochs):
        total = 0
        while total<train_pairs.shape[0]:
            node_pairs = train_data[total:total+64]
            src_f_fea,src_g_fea,trg_f_fea,trg_g_fea = emb_model.forward(src_fea,trg_fea)

            optimizer.zero_grad()
            l = emb_model.loss(src_g_fea,src_f_fea,trg_g_fea,trg_f_fea, node_pairs)
            l.backward()
            optimizer.step()
            total+=64

    """
    for ep in range(args.fea_epochs):
        total = 0
        while total<train_pairs.shape[0]:
            optimizer.zero_grad()
            node_pairs = train_data[total:total+64]
            src_f_fea,src_g_fea,trg_f_fea,trg_g_fea = emb_model.forward(src_fea,trg_fea)
            l = emb_model.loss(src_g_fea,src_f_fea,trg_g_fea,trg_f_fea, node_pairs)
            l.backward()
            optimizer.step()
            total+=64
    """

    f1,_,f2,_ = emb_model.forward(src_fea,trg_fea)
    return f1.cpu().detach().numpy(),f2.cpu().detach().numpy()




if __name__=='__main__':
    pass