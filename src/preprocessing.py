import numpy as np
import scipy.sparse as sp

# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and
# https://github.com/tkipf/gcn
# https://github.com/vmasrani/gae_in_pytorch
# My thanks to them!
# ------------------------------------


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0]) # Add the self-loops back in
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def mask_test_edges(adj, test_percent=10., val_percent=20.):

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    if test_percent != 0. and val_percent != 0:

        num_test = int(np.floor(edges.shape[0] / test_percent))
        num_val = int(np.floor(edges.shape[0] / val_percent))

        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                    np.all(np.any(rows_close, axis=0), axis=0))

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(val_edges_false, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)

        data = np.ones(train_edges.shape[0])

        # Re-build adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

    else:
        adj_train = adj + adj.T
        train_edges = edges_all

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def mask_edges(adj, test_edges, val_edges):
    """Given a test of edges, remove them from a graph"""

    edges_to_remove = np.append(test_edges, val_edges, axis=0)

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    edge_set = set(map(tuple, edges.tolist()))
    edge_rm_set = set(map(tuple, edges_to_remove.tolist()))
    train_edges = np.array(list(edge_set - edge_rm_set))

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges