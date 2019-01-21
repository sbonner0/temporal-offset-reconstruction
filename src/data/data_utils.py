import csv
import os
import pickle as pkl

# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and
# https://github.com/tkipf/gcn
# https://github.com/vmasrani/gae_in_pytorch
# My thanks to them!
# ------------------------------------

import networkx as nx
import numpy as np
import scipy.sparse as sp
from graph_tool.all import *
from pylab import *
from sklearn import preprocessing


def printDegreeDist(g, name):
    # Plot the degree distribution of the passed graph
    total_hist = vertex_hist(g, "total")
    y = total_hist[0]
    figure(figsize=(6,4))
    errorbar(total_hist[1][:-1], total_hist[0], fmt="o", label="total")
    gca().set_yscale("log")
    gca().set_xscale("log")
    subplots_adjust(left=0.2, bottom=0.2)
    xlabel("$k_{total}$")
    ylabel("$NP(k_{total})$")
    #tight_layout()
    savefig(name)

def randomRewrite(tempG, statModel, num_edges_to_rewire):
    # Method to rewire the graph based on some probabilistic methods.
    # Does not increase or decrease the number of vertices or edges.
    # Will rewire the graph in place
    #https://graph-tool.skewed.de/static/doc/generation.html#graph_tool.generation.random_rewire

    random_rewire(tempG, model=statModel, n_iter=num_edges_to_rewire, edge_sweep=False)

    return tempG

def load_adj_graph(data_location):
    """Load a given graph and return the adjacency matrix"""

    adj = sp.load_npz(data_location)
    return adj, None

def load_features(data_location):
    """Load a given graph and return the adjacency matrix"""

    features = sp.load_npz(data_location)
    return features

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
        with open("ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "ind.{}.test.index".format(dataset))
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

def save_gt_graph(graph, filename):
    """Save graph_tool graph as npz file"""

    adj = adjacency(graph)
    sp.save_npz(filename, adj)

def save_gt_features(features, filename):
    """Save graph_tool features as npz file"""

    features = sp.csr_matrix(features)
    sp.save_npz(filename, features)


def rewire_and_save(dataset, num_of_rewires=5, print_deg_dist=False):
    """Load a given graph and rewire it a given number of times"""

    # Load the cora or citeseer datasets
    adj, _ = load_data(dataset)
    # Transform into matrix into a graph_tool graph
    g = Graph()
    g.add_edge_list(np.array([line.split(",") for line in nx.generate_edgelist(nx.from_numpy_matrix(np.array(adj.todense())), delimiter=',', data=False)]), hashed=True)
    g.set_directed(False)
    save_gt_graph(g, f'{dataset}.npz')

    # Rewire the graph N times
    # Changing the number of rewired edges helps our approach
    num_rewire_edges = int(g.num_edges())
    
    for i in range(num_of_rewires):
        print(g)
        #g = randomRewrite(g, 'constrained-configuration', num_rewire_edges)
        g = randomRewrite(g, 'erdos', num_rewire_edges)
        save_gt_graph(g, f'{dataset}_{i}_rewire.npz')
        if print_deg_dist: 
            printDegreeDist(g, f'{i}_top.png')


def extract_vertex_fingerprint(temp_graph):
    """Extract the vertex features from a graph and return normalised array"""

    # Extract vertex features via graph-tool
    temp_graph.vertex_properties["dp"] = temp_graph.degree_property_map("total")
    temp_graph.vertex_properties["lc"] = local_clustering(temp_graph)
    temp_graph.vertex_properties["pR"] = pagerank(temp_graph)
    temp_graph.vertex_properties["eV"] = eigenvector(temp_graph)[1]
    temp_graph.vertex_properties["vbtwn"], _ = betweenness(temp_graph)
    #temp_graph.vertex_properties["clos"] = closeness(temp_graph)
    print("Vertex feature extraction complete")

    # Put all features into matrix and normalise
    dp = np.array(temp_graph.vp["dp"].a)
    lc = np.array(temp_graph.vp["lc"].a)
    pR = np.array(temp_graph.vp["pR"].a)
    eV = np.array(temp_graph.vp["eV"].a)
    vbtwn = np.array(temp_graph.vp["vbtwn"].a)
    #clos = np.array(temp_graph.vp["clos"].a)

    scaled_features = preprocessing.scale(np.concatenate((dp[:,np.newaxis], lc[:,np.newaxis], pR[:,np.newaxis], eV[:,np.newaxis], vbtwn[:,np.newaxis]), axis=1), axis=1)
    
    return scaled_features

def process_empirical_temporal_graphs(edges_filepath="out.ca-cit-HepTh", num_time_splits=100):
    """Function to load and process empirical temporal graphs"""
    
    # Load the edges from file
    full_graph = load_graph_from_csv(edges_filepath, directed=False, hashed=True, eprop_types=['int', 'int'], eprop_names=['w', 'date'], csv_options={'delimiter': ' '})
    print("Graph Loaded From Disk")

    # Get the range of dates from the edges
    dates = []
    for e in full_graph.edges():
        dates.append(full_graph.ep.date[e])
    dates = np.array(dates)
    date_ranges = np.linspace(dates.min(), dates.max(), num=num_time_splits) # Splits the timestamps into linearly separated ranges. 
    print("Timestamps Loaded")

    # Filter the graph
    graph_counter = 0
    for i in range(10, 101, 10):
        if i >= 40:
            temp_graph = GraphView(full_graph, efilt=lambda e: full_graph.ep.date[e] <= int(date_ranges[i-1]))
            assert full_graph.num_vertices() == temp_graph.num_vertices() # Check both graphs have the required number of vertices
            save_gt_graph(temp_graph, f'cit-hepth_{graph_counter}_rewire.npz')
            sacled_features = extract_vertex_fingerprint(temp_graph)
            save_gt_features(sacled_features, f'cit-hepth_{graph_counter}_features.npz')       
            graph_counter += 1
            print(temp_graph)

if __name__ == "__main__":

    #load_edgelist_graph("graph.txt")
    rewire_and_save('cora', 5)
    rewire_and_save('citeseer', 5)
    #process_empirical_temporal_graphs()
