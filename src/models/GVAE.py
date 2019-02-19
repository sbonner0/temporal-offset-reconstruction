from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution

class GVAE(nn.Module):
    """Graph Auto Encoder (see: https://arxiv.org/abs/1611.07308) - Variational Version"""

    def __init__(self, data, n_hidden, n_latent, dropout, bias, xavier_init=True):
        super().__init__()

        # Device
        self.device = None

        # Data
        self.x = data['features']
        self.adj_norm = data['adj_norm']
        self.adj_labels = data['adj_labels']    

        # Dimensions
        N, D = data['features'].shape
        self.n_samples = N
        self.n_edges = self.adj_labels.sum()
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.bias = bias

        # Parameters
        self.pos_weight = float(N * N - self.n_edges) / self.n_edges
        self.norm = float(N * N) / ((N * N - self.n_edges) * 2)

        self.gc1 = GraphConvolution(self.input_dim, self.n_hidden, self.bias)
        self.gc2_mu = GraphConvolution(self.n_hidden, self.n_latent, self.bias)
        self.gc2_sig = GraphConvolution(self.n_hidden, self.n_latent, self.bias)
        self.dropout = dropout

        if xavier_init:
            # Initialise the GCN weights to Xavier Uniform
            torch.nn.init.xavier_uniform_(self.gc1.weight)
            torch.nn.init.xavier_uniform_(self.gc2_mu.weight)
            torch.nn.init.xavier_uniform_(self.gc2_sig.weight)

    def encode_graph(self, x, adj):
        # First layer shared between mu/sig layers
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        self.mu = self.gc2_mu(x, adj)
        self.log_sig = self.gc2_sig(x, adj)

        self.z = self.mu + (torch.randn(self.n_samples, self.n_latent, device=self.device) * torch.exp(self.log_sig)) #.to(self.device) 

        return self.z

    def decode_graph(self, z):
        # Here the reconstruction is based upon the inner product between the latent representation
        adj_hat = torch.mm(z, z.t())

        return adj_hat

    def get_embeddings(self, x, adj):

        return self.encode_graph(x, adj)

    def forward(self, x, adj):
        # Encode and then decode the graph
        x_hat = self.encode_graph(x, adj)
        adj_hat = self.decode_graph(x_hat)

        return adj_hat
