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
    #output = -2*torch.bmm(x, x.t()) 
    #print(x.shape)
    #output += F.normalize(instances_norm )
    #print(instance_norm.shape)
    #output += instances_norm.t()

    output = -2*torch.mm(x, x.t()) 
    output += instances_norm 
    output += instances_norm.t()
    return torch.sqrt(output.clamp(min=0) + eps)




def dcor(x, y):
    eps=1e-9
    m,_ = x.shape
    #print(x.shape, y.shape)

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

    return dcor





class CANA_fea_1(nn.Module):
    def __init__(self, in_dim):
        super(CANA_fea_1, self).__init__()
        self.p = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            )

        self.q = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            )

    def forward(self, src_fea, trg_fea):
        src_p_fea = F.normalize(self.p(src_fea))
        src_q_fea = F.normalize(self.q(src_fea))

        trg_p_fea = F.normalize(self.p(trg_fea))
        trg_q_fea = F.normalize(self.q(trg_fea))

        return src_p_fea,src_q_fea,trg_p_fea,trg_q_fea

    def loss(self,src_p_fea,src_q_fea,trg_p_fea,trg_q_fea,node_pairs):
        pair_num = node_pairs.shape[0]
        f1 = F.normalize(src_p_fea[node_pairs[:,0]])
        f2 = F.normalize(trg_p_fea[node_pairs[:,1]])
        neg_f2 = f2[torch.randperm(pair_num)]
        #neg_f3 = f2[torch.randperm(pair_num)]

        loss_anchor=torch.sum(torch.mul(f1,f2),1)-torch.sum(torch.mul(f1,neg_f2),1)#-torch.sum(torch.mul(f1,neg_f3),1)
        loss_anchor = -torch.mean(loss_anchor)

        p_p1 = torch.mean(src_q_fea,0)
        p_p2 = torch.mean(trg_q_fea,0)
        loss_center=-torch.norm(p_p1-p_p2)
        
        loss_dcor = (1/src_p_fea.shape[0]**2)*dcor(src_p_fea,src_q_fea)+(1/trg_p_fea.shape[0]**2)*dcor(trg_p_fea,trg_q_fea)

        return loss_anchor+loss_center+loss_dcor




class CANA_nei_1(nn.Module):
    def __init__(self, in_dim, args, device, src_ppmi, trg_ppmi):
        super(CANA_nei_1, self).__init__()
        self.hidden_dim = args.emb_dim
        self.layers = args.layers
        self.device = device
        hid1 = args.emb_dim

        self.src_ppmi = src_ppmi
        self.trg_ppmi = trg_ppmi

        self.fx = nn.Sequential(nn.Linear(in_dim,hid1),
            nn.GELU(),
            nn.Linear(hid1,hid1),
            nn.GELU())
        
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
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,1),
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

            edge_weight_c = (edge_att+hard_src)/2#+hard_src
            edge_weight_o = 1-edge_weight_c

            x = self.act(conv(x, src_g, edge_weight_c))
            cfd = self.act(conv(x, src_g, edge_weight_o))

            embs_list.append(x)
            confd_src.append(cfd)

        x = trg_fea1
        for i, conv in enumerate(self.convs):
            edge_rep = torch.cat([x[trg_row], x[trg_col]], dim=-1)
            edge_att = torch.sigmoid(self.edge_att_mlp(edge_rep))

            edge_weight_c = (edge_att+hard_trg)/2#+hard_src
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
    


if __name__=='__main__':
    pass