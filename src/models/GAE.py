from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution

class GAE(nn.Module):
    """Graph Auto Encoder (see: https://arxiv.org/abs/1611.07308) - Probabilistic Version"""

    def __init__(self, data, n_hidden, n_latent, dropout, bias, xavier_init=True):
        super().__init__()

        # Data
        self.x = data['features']
        self.adj_norm = data['adj_norm']
        self.adj_labels = data['adj_labels']    

        # Dimensions
        N, D = data['features'].shape
        self.n_edges = self.adj_labels.sum()
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.bias = bias

        # Parameters
        self.pos_weight = float(N * N - self.n_edges) / self.n_edges
        self.norm = float(N * N) / ((N * N - self.n_edges) * 2)

        self.gc1 = GraphConvolution(self.input_dim, self.n_hidden, self.bias)
        self.gc2 = GraphConvolution(self.n_hidden, self.n_latent, self.bias)
        self.dropout = dropout

        # Adding PReLU seemingly made a difference for TO
        self.prelu1 = nn.PReLU()

        if xavier_init:
            # Initialise the GCN weights to Xavier Uniform
            torch.nn.init.xavier_uniform_(self.gc1.weight)
            torch.nn.init.xavier_uniform_(self.gc2.weight)

    def encode_graph(self, x, adj):

        # Perform the encoding stage using a two layer GCN
        x = self.prelu1(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x

    def forward(self, x, adj):

        # Encoder
        x = self.encode_graph(x, adj)
        # Decoder
        x = F.dropout(x, self.dropout, training=self.training)
        adj_hat = torch.spmm(x, x.t())

        return adj_hat

    def get_embeddings(self, x, adj):

        return self.encode_graph(x, adj)