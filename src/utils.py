import csv
import pickle as pkl
import sys

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)

from data.data_utils import load_adj_graph, load_features
from preprocessing import mask_edges, mask_test_edges, preprocess_graph

# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and
# https://github.com/tkipf/gcn
# https://github.com/vmasrani/gae_in_pytorch
# My thanks to them!
# ------------------------------------


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def eval_gae(edges_pos, edges_neg, emb):
    """Evaluate the GAE model via link prediction"""
    # TODO: This seems wastfull to do every epoch

    # Predict on test set of edges
    emb = emb.cpu()
    adj_rec = torch.sigmoid(torch.spmm(emb, emb.t())).cpu().numpy()
    preds = []
    
    # Loop over the positive test edges
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
    
    preds_neg = []

    # Loop over the negative test edges
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

    accuracy = accuracy_score(labels_all, (preds_all > 0.5).astype(float))
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return accuracy, roc_score, ap_score


def make_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def plot_results(results, test_freq, path='results.png'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_elbo']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results['train_elbo'])
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend(['Train'], loc='upper right')

    # Accuracy
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_train, results['accuracy_train'])
    ax.plot(x_axis_test, results['accuracy_test'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend(['Train', 'Test'], loc='lower right')

    # ROC
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, results['roc_train'])
    ax.plot(x_axis_test, results['roc_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Precision
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x_axis_train, results['ap_train'])
    ax.plot(x_axis_test, results['ap_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Save
    fig.tight_layout()
    fig.savefig(path)

def log_tensorboard(dict, global_step, writer):
    """Helper function to allow easy use of Tensorboard"""

    for name in dict:
        if isinstance(dict[name], torch.Variable):
            dict[name] = dict[name].data
        if isinstance(dict[name], torch.Tensor):
            dict[name] = dict[name].mean()
        writer.add_scalar(name,dict[name], global_step=global_step)


def save_model(epoch, model, optimizer, filepath="model.cpt"):
    """Save the model and embeddings"""

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state, filepath)
    print("Model Saved")

def evaluate_on_testset(model, test_edges, test_edges_false, data):
    """Evaluate the model on a given test set of graph edges"""

    with torch.no_grad():
        model.eval()
        emb = model.get_embeddings(data['features'], data['adj_norm'])
        accuracy, roc_score, ap_score = eval_gae(test_edges, test_edges_false, emb)
    model.train()

    return accuracy, roc_score, ap_score


def eval_on_new_edges_only(model, test_edges, test_edges_false, train_edges, data):
    """Evaluate the model on only edges not in the original graph"""

    edge_set = set(map(tuple, test_edges.tolist())) # Convert the list of edges to a set
    edge_rm_set = set(map(tuple, train_edges.tolist())) 
    edges_pos = np.array(list(edge_set - edge_rm_set))

    edge_set_neg = set(map(tuple, test_edges_false)) 
    edges_neg = np.array(list(edge_set_neg - edge_rm_set))

    #temp = int(edges_pos.size/2)
    #edges_neg = edges_neg[:temp, :] 

    with torch.no_grad():
        model.eval()
        emb = model.get_embeddings(data['features'], data['adj_norm'])
        adj_rec = torch.sigmoid(torch.mm(emb, emb.t())).cpu().numpy()

        preds = []

        # Loop over the positive test edges
        for e in edges_pos:
            preds.append(adj_rec[e[0], e[1]])
        
        preds_neg = []

        # Loop over the negative test edges
        for e in edges_neg:
            preds_neg.append(adj_rec[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

        accuracy = accuracy_score(labels_all, (preds_all > 0.5).astype(float))
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)


    return accuracy, roc_score, ap_score

def evaluate_model(args, gae, data, device, train_edges):
    """The evaluation code for the model"""

    print(f"Using the evaluation mode {args.test_mode}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset_str == "cit-hepth":
        idx_range = (1, 7)
    else:
        idx_range = (0, 5)

    if args.test_mode == "fixed":
        # Setup CSV file to process results
        with open(f'results/{args.name}_{args.dataset_str}.csv', "w") as csv_f:
            csv_file = csv.writer(csv_f, delimiter=",", lineterminator="\n")

            # Loop over the number of temporal test graphs
            for i in range(idx_range[0], idx_range[1]):

                print(f'Running on temporal graph {i}')
                time_adj, time_features = load_adj_graph(f'data/{args.dataset_str}_{i}_rewire.npz')    

                # Store original adjacency matrix (without diagonal entries)
                time_adj_orig = time_adj
                time_adj_orig = time_adj_orig - sp.dia_matrix((time_adj_orig.diagonal()[np.newaxis, :], [0]), shape=time_adj_orig.shape)
                time_adj_orig.eliminate_zeros()

                acu_arr = []
                roc_arr = []
                ap_arr = []

                new_acu_arr = []
                new_roc_arr = []
                new_ap_arr = []

                # Create new random splits of data and present results as the mean of this
                for r in range(10):
                    
                    # Set the seed to I for a new, but repeatable, split
                    np.random.seed(r)

                    # Get the test score on the full set of edges
                    _, _, _, _, test_edges, test_edges_false = mask_test_edges(time_adj, 80., 10.)
                    accuracy, roc_score, ap_score = evaluate_on_testset(gae, test_edges, test_edges_false, data)
                    acu_arr.append(accuracy)
                    roc_arr.append(roc_score)
                    ap_arr.append(ap_score)

                    # Get the test score on the new edges only
                    accuracy, roc_score, ap_score = eval_on_new_edges_only(gae, test_edges, test_edges_false, train_edges, data)
                    new_acu_arr.append(accuracy)
                    new_roc_arr.append(roc_score)
                    new_ap_arr.append(ap_score)


                print(f'Test Accuracy: {np.mean(acu_arr):.4f} +/- {np.std(acu_arr):.4f}')
                print(f'Test ROC score: {np.mean(roc_arr):.4f} +/- {np.std(roc_arr):.4f}')
                print(f'Test AP score: {np.mean(ap_arr):.4f} +/- {np.std(ap_arr):.4f}')

                print("Scores on new edges only:")
                print(f'Test Accuracy: {np.mean(new_acu_arr):.4f} +/- {np.std(new_acu_arr):.4f}')
                print(f'Test ROC score: {np.mean(new_roc_arr):.4f} +/- {np.std(new_roc_arr):.4f}')
                print(f'Test AP score: {np.mean(new_ap_arr):.4f} +/- {np.std(new_ap_arr):.4f}')
                print("")

                csv_file.writerow([args.name, args.dataset_str, i, np.mean(roc_arr), np.std(roc_arr), np.mean(ap_arr), np.std(ap_arr), np.mean(new_roc_arr), np.std(new_roc_arr), np.mean(new_ap_arr), np.std(new_ap_arr)])

    elif args.test_mode == "change":
    # Different evaluation code -------------------------------------------------------------------------------------

        org_train_edges = train_edges

        # Setup CSV file to process results
        with open(f'results/change_{args.name}_{args.dataset_str}.csv', "w") as csv_f:
            csv_file = csv.writer(csv_f, delimiter=",", lineterminator="\n")

            # Loop over the number of temporal test graphs
            for i in range(idx_range[0], idx_range[1]):

                print(f'Running on temporal graph {i}')
                acu_arr = []
                roc_arr = []
                ap_arr = []
                new_acu_arr = []
                new_roc_arr = []
                new_ap_arr = []

                # Create new random splits of data and present results as the mean of this
                for r in range(10):
                    
                    # Set the seed to I for a new, but repeatable, split
                    np.random.seed(r)

                    adj, _ = load_adj_graph(f'data/{args.dataset_str}_{i}_rewire.npz')

                    # Store original adjacency matrix (without diagonal entries)
                    adj_orig = adj
                    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

                    # Some preprocessing
                    adj_train_norm   = preprocess_graph(adj_train)
                    adj_train_norm   = make_sparse(adj_train_norm)
                    adj_train_labels = torch.FloatTensor(adj_train + sp.eye(adj_train.shape[0]).todense())
                    
                    if args.use_features == 'n':
                        
                        features = sp.eye(adj_train.shape[0]).tolil()
                    else:
                        
                        features = load_features(f'data/{args.dataset_str}_{i}_features.npz').tolil()

                    features = make_sparse(features)
                    
                    data = {
                        'adj_norm'  : adj_train_norm,
                        'adj_labels': adj_train_labels,
                        'features'  : features,
                    }

                    data['adj_norm'] = data['adj_norm'].to(device)
                    data['adj_labels'] = data['adj_labels'].to(device)
                    data['features'] = data['features'].to(device)

                    # Get the test score on the full set of edges
                    accuracy, roc_score, ap_score = evaluate_on_testset(gae, test_edges, test_edges_false, data)
                    acu_arr.append(accuracy)
                    roc_arr.append(roc_score)
                    ap_arr.append(ap_score)

                    # Get the test score on the new edges only
                    accuracy, roc_score, ap_score = eval_on_new_edges_only(gae, test_edges, test_edges_false, org_train_edges, data)
                    new_acu_arr.append(accuracy)
                    new_roc_arr.append(roc_score)
                    new_ap_arr.append(ap_score)


                print(f'Test Accuracy: {np.mean(acu_arr):.4f} +/- {np.std(acu_arr):.4f}')
                print(f'Test ROC score: {np.mean(roc_arr):.4f} +/- {np.std(roc_arr):.4f}')
                print(f'Test AP score: {np.mean(ap_arr):.4f} +/- {np.std(ap_arr):.4f}')

                print("Scores on new edges only:")
                print(f'Test Accuracy: {np.mean(new_acu_arr):.4f} +/- {np.std(new_acu_arr):.4f}')
                print(f'Test ROC score: {np.mean(new_roc_arr):.4f} +/- {np.std(new_roc_arr):.4f}')
                print(f'Test AP score: {np.mean(new_ap_arr):.4f} +/- {np.std(new_ap_arr):.4f}')
                print("")

                csv_file.writerow([args.name, args.dataset_str, i, np.mean(roc_arr), np.std(roc_arr), np.mean(ap_arr), np.std(ap_arr), np.mean(new_roc_arr), np.std(new_roc_arr), np.mean(new_ap_arr), np.std(new_ap_arr)])
