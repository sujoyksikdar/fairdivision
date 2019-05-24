###########################################################
# nw(V, A) # compute Nash welfare of A under valuations V
# construct_alloc_graph(V, A)
# max_cardinality_allocation(V)
# max_match_allocation(V)
# get_halls_instance(V)
# get_solution_2_original(Xprime, V, matched)
###########################################################

import numpy as np
import networkx as nx


def nw(V, A):
    (n, m) = np.shape(V)
    U = list()
    nw = 0
    for j in range(n):
        u = np.dot(V[j, :], A[j, :])
        U.append(u)
    if not np.all(np.array(U) == 0):
        nw = np.prod([np.float64(u) for u in U if u > 0])
    return nw, U


def construct_alloc_graph(V, A):
    (n, m) = np.shape(V)
    nodes = list()
    nodes = nodes + [j for j in range(n)] # add nodes for agents
    edges = list()
    edge_labels = dict()
    # add allocation edges
    for j in range(n):
        for k in range(n):
            for i in range(m):
                if V[j, i] == 1 and A[k, i] == 1:
                    edges.append((j, k))
                    edge_labels[(j, k)] = i
                    break
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G, edge_labels


def max_cardinality_allocation(V):
    (n, m) = np.shape(V)
    G = nx.Graph()
    agent_nodes = [('a', j) for j in range(n)]
    G.add_nodes_from(agent_nodes, bipartite=0)
    item_nodes = [('g', i) for i in range(m)]
    G.add_nodes_from(item_nodes, bipartite=1)
    edges = list()
    for j in range(n):
        for i in range(m):
            if V[j, i] > 0:
                edges.append((('a', j), ('g', i)))
    G.add_edges_from(edges)
    top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    M = nx.bipartite.maximum_matching(G, top_nodes=top_nodes)
    A = np.zeros((n, m))
    for match in M:
        if match[0] == 'a':
            item = M[match]
            j = match[1]
            i = item[1]
            A[j, i] = 1
    for i in range(m):
        is_allocated = (np.sum(A[:, i]) > 0)
        if is_allocated:
            continue
        j = np.argmax(V[:, i])
        A[j, i] = 1
    return A


def max_match_allocation(V):
    (n, m) = np.shape(V)
    G = nx.Graph()
    agent_nodes = [('a', j) for j in range(n)]
    G.add_nodes_from(agent_nodes, bipartite=0)
    item_nodes = [('g', i) for i in range(m)]
    G.add_nodes_from(item_nodes, bipartite=1)
    edges = list()
    for j in range(n):
        for i in range(m):
            if V[j, i] == np.max(V[:, i]):
                edges.append((('a', j), ('g', i)))
    G.add_edges_from(edges)
    top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    M = nx.bipartite.maximum_matching(G, top_nodes=top_nodes)
    A = np.zeros((n, m))
    for match in M:
        if match[0] == 'a':
            item = M[match]
            j = match[1]
            i = item[1]
            A[j, i] = 1
    for i in range(m):
        is_allocated = (np.sum(A[:, i]) > 0)
        if is_allocated:
            continue
        j = np.argmax(V[:, i])
        A[j, i] = 1
    return A


def get_halls_instance(V):
    (n, m) = np.shape(V)
    A = max_cardinality_allocation(V)
    matched = [j for j in range(n) if np.sum(A[j, :]) > 0]
    Vprime = V[matched, :]
    return Vprime, matched


def recover_from_halls(Xprime, V, matched):
    (n, m) = np.shape(V)
    X = np.zeros((n, m))
    num_matched = len(matched)
    for k in range(num_matched):
        j = matched[k]
        X[j, :] = Xprime[k, :]
    return X


def get_valued_instance(V):
    (n, m) = np.shape(V)
    valued = [i for i in range(m) if np.sum(V[:, i]) > 0]
    Vprime = V[:, valued]
    return Vprime, valued


def recover_from_valued(Xprime, V, valued):
    (n, m) = np.shape(V)
    X = np.zeros((n, m))
    num_valued = len(valued)
    for h in range(num_valued):
        i = valued[h]
        X[:, i] = Xprime[:, h]
    for i in range(m):
        if i not in valued:
            X[0, i] = 1
    return X


def mat2set(A):
    alloc = dict()
    n = len(A)
    for i in range(n):
        alloc[i] = A[i].nonzero()[0]
    return alloc


def compute_utilities(V, A):
    (n, m) = np.shape(V)
    U = np.zeros((n, m))  # U[j, k] = utility of agent j for agent k's bundle
    for j in range(n):
        for k in range(n):
            U[j, k] = np.dot(V[j, :], A[k, :])
    return U
