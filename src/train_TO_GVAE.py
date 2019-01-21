import argparse
import csv
import datetime
import os
import time
from collections import defaultdict

# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and
# https://github.com/tkipf/gcn
# https://github.com/vmasrani/gae_in_pytorch
# My thanks to them!
# ------------------------------------

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from data.data_utils import load_adj_graph, load_features
from models.GVAE import GVAE
from preprocessing import mask_edges, mask_test_edges, preprocess_graph
from utils import (dotdict, eval_gae, eval_on_new_edges_only,
                   evaluate_on_testset, load_data, make_sparse, plot_results,
                   save_model, evaluate_model)


def main(args):
    """ Train GVAE Via Temporal Off-Set Reconstruction """ 

    # Compute the device upon which to run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # TensorboardX logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    log_dir = os.path.join(args.logs_path, args.name + '_' + timestamp)
    writer = SummaryWriter(log_dir = log_dir)

    print("Train GVAE Via Temporal Off-Set Reconstruction")
    print(f'Using {args.dataset_str} dataset')

    # Load data
    adj, features = load_adj_graph(f'data/{args.dataset_str}.npz') # Load the original graph
    time_adj, time_features = load_adj_graph(f'data/{args.dataset_str}_0_rewire.npz')

    # Store original adjacency matrix (without diagonal entries)
    time_adj_orig = time_adj
    time_adj_orig = time_adj_orig - sp.dia_matrix((time_adj_orig.diagonal()[np.newaxis, :], [0]), shape=time_adj_orig.shape)
    time_adj_orig.eliminate_zeros()

    # Load the data from the training graph
    adj_train, train_edges, _, _, _, _ = mask_test_edges(adj, test_percent=0., val_percent=0.)
    adj_train = adj
    adj_train = adj_train - sp.dia_matrix((adj_train.diagonal()[np.newaxis, :], [0]), shape=adj_train.shape)
    adj_train.eliminate_zeros()

    # Load the data from the temporally off-set target graph
    time_adj_train, _, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(time_adj)

    # Remove the test edges from the training graph
    adj_train, _ = mask_edges(adj_train, test_edges, val_edges)

    # Check the two graphs are the same size
    assert adj_train.shape == time_adj_train.shape

    # Some preprocessing
    adj_train_norm   = preprocess_graph(adj_train)
    adj_train_norm   = make_sparse(adj_train_norm)
    adj_train_labels = torch.FloatTensor(time_adj_train + sp.eye(time_adj_train.shape[0]).todense())

    if args.use_features == 'n':
        print("NO Vertex Features")
        features = sp.eye(adj_train.shape[0]).tolil()
    else:
        print("Vertex Features Used")
        features = load_features(f'data/{args.dataset_str}_0_features.npz').tolil()

    features = make_sparse(features)
    N, _ = adj_train.shape
    
    data = {
        'adj_norm'  : adj_train_norm,
        'adj_labels': adj_train_labels,
        'features'  : features,
    }

    gae = GVAE(data,
              n_hidden=args.hidden_size1,
              n_latent=args.hidden_size2,
              dropout=args.dropout,
              bias=args.gcn_bias,
              xavier_init=True)

    # Send the model and data to the available device
    gae.to(device)
    gae.norm = torch.as_tensor(gae.norm).to(device)
    gae.device = device
    data['adj_norm'] = data['adj_norm'].to(device)
    data['adj_labels'] = data['adj_labels'].to(device)
    data['features'] = data['features'].to(device)

    optimizer = optim.Adam(gae.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=args.weight_decay)
    
    # Results
    results = defaultdict(list)
    
    # Full batch training loop
    for epoch in range(args.num_epochs):

        t = time.time()
        gae.train()
        optimizer.zero_grad()

        # forward pass
        output = gae(data['features'], data['adj_norm'])

        # Compute the loss ------------------------------------------

        # Compute the weighted_cross_entropy_with_logits ------------
        logits = output
        targets = data['adj_labels']
        loss = gae.norm * F.binary_cross_entropy_with_logits(logits, targets, pos_weight=gae.pos_weight)

        # compute the latent loss -----------------------------------
        # Two possible ways of computing KL - second seems to work best!
        #kl = (0.5 / N) * torch.mean( torch.sum( 1 + 2 * gae.log_sig - gae.mu.pow(2) - torch.exp(gae.log_sig).pow(2), 1) )
        kl = (0.5 / N) * torch.mean( torch.sum( 1 + gae.log_sig - gae.mu.pow(2) - torch.exp(gae.log_sig), 1) )

        # Subtract the KL from the loss
        loss -= kl

        correct_prediction = logits.ge(0.5).int().eq(targets.int())
        training_accuracy = torch.mean(correct_prediction.float())
        
        loss.backward()
        optimizer.step()

        results['train_elbo'].append(loss.item())

        # Evaluation step ----------------------------------------------------
        accuracy, roc_score, ap_score = evaluate_on_testset(gae, val_edges, val_edges_false, data)
        results['accuracy_train'].append(accuracy)
        results['roc_train'].append(roc_score)
        results['ap_train'].append(ap_score)
            
        print(f'Epoch: {(epoch + 1):4} train_loss= {loss.item():.5f} train_accuracy= {training_accuracy:.5f} val_acc= {accuracy:.5f} val_roc= {roc_score:.5f} vap_ap= {ap_score:.5f} | time_taken={time.time()-t:.2f} (secs)')

        # Test loss
        if epoch % args.test_freq == 0:
            accuracy, roc_score, ap_score = evaluate_on_testset(gae, test_edges, test_edges_false, data)
            results['accuracy_test'].append(accuracy)
            results['roc_test'].append(roc_score)
            results['ap_test'].append(ap_score)
    
    print("Optimization Finished!")
    
    # -------------- Compute metrics on the test set --------------
    accuracy, roc_score, ap_score = evaluate_on_testset(gae, test_edges, test_edges_false, data)
    print(f'Test Accuracy: {accuracy}')
    print(f'Test ROC score: {roc_score}')
    print(f'Test AP score: {ap_score}')

    # Save the model
    save_model(args.num_epochs, gae, optimizer)
    
    # Plot
    plot_results(results, args.test_freq, path= args.dataset_str + "_GAE_results.png")

    # -------------- Predict edges on a third temporal graph --------------
    print(" ----------- Testing on Temporal Graphs ----------- ")
    evaluate_model(args, gae, data, device, train_edges)

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--name', type=str, default='TO_GVAE',
                        help='Model name.')  
    parser.add_argument('--dataset_str', type=str, default='cora',
                        help='Dataset string.')  
    parser.add_argument('--logs_path', type=str, default='/tmp/',
                        help='Use vertex features.')  
    parser.add_argument('--test_freq', type=int, default=10,
                        help='How often to run the test set.')

    parser.add_argument('--test_mode', type=str, default='change',
                        help='What eval mode to use.')

    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_size1', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--hidden_size2', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--gcn_bias', type=bool, default=True,
                        help='If to use bias terms.')
    parser.add_argument('--use_features', type=str, default='n',
                        help='Use vertex features or not.')                           

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)