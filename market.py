import numpy as np
import networkx as nx
from copy import deepcopy

from solution_util import *


def round_instance(V):
    (n, m) = np.shape(V)
    vmax = np.max(V)
    eps = 1/(6 * (np.float64(m) ** 3) * vmax)
    Vprime = np.zeros((n, m))
    for j in range(n):
        for i in range(m):
            if V[j, i] > 0:
                Vprime[j, i] = (1 + eps) ** np.ceil(np.log(V[j, i]) / np.log(1 + eps))
    return Vprime, eps


def is3epef1(X, p, eps):
    (n, m) = np.shape(X)
    spending = np.zeros(n)
    for j in range(n):
        spending[j] = np.dot(p, X[j, :])
    min_spend = np.min(spending) # least spend by any agent
    for j in range(n):
        if np.sum(X[j, :]) <= 1:
            continue
        vals = [spending[j] - (p[i] * X[j, i]) for i in range(m) if X[j, i] > 0]
        if np.all(np.array(vals) > (1 + 3*eps) * min_spend):
            return False
    return True


def market_phase1(V, eps):
    print('in phase 1')
    (n, m) = np.shape(V)
    X = np.zeros((n, m))
    p = np.zeros(m)
    for i in range(m):
        ivals = V[:, i]
        j = np.argmax(ivals)
        X[j, i] = 1
        p[i] = V[j, i]
    if is3epef1(X, p, eps):
        return True, X, p
    return False, X, p


def bpb(V, p):
    (n, m) = np.shape(V)
    BB = np.zeros((n, m)) # bang per buck
    for j in range(n):
        for i in range(m):
            BB[j, i] = V[j, i] / p[i]
    return BB


def build_mbb_graph(V, p):
    (n, m) = np.shape(V)
    BB = bpb(V, p)
    MBB = list()
    MBB_edges = list()
    for j in range(n):
        MBB_items_j = np.argwhere(BB[j, :] == np.amax(BB[j, :])).flatten().tolist()
        MBB.append(MBB_items_j)
        MBB_edges = MBB_edges + [(('a', j), ('g', i)) for i in MBB_items_j]
    G_MBB = nx.DiGraph()
    nodes = [('a', j) for j in range(n)]
    nodes = nodes + [('g', i) for i in range(m)]
    G_MBB.add_nodes_from(nodes)
    G_MBB.add_edges_from(MBB_edges)
    return G_MBB, MBB_edges, MBB


def augment_mbb_graph(G_MBB, X):
    (n, m) = np.shape(X)
    X_edges = list()
    for j in range(n):
        for i in range(m):
            if X[j, i] > 0:
                X_edges.append((('g', i), ('a', j)))
    G_Augmented = deepcopy(G_MBB)
    G_Augmented.add_edges_from(X_edges)
    return G_Augmented, X_edges


def build_augmented_mbb_graph(V, X, p):
    G_MBB, MBB_edges, MBB = build_mbb_graph(V, p)
    G_Augmented, X_edges = augment_mbb_graph(G_MBB, X)
    return G_Augmented


def build_hierarchy(j, G, X):
    (n, m) = np.shape(X)
    H = dict()
    for l in range(n):
        H[l] = list()
    for k in range(n):
        try:
            l = nx.shortest_path_length(G, source=('a', j), target=('a', k))/2
            H[l].append(k)
        except nx.NetworkXNoPath:
            continue
    return H


def market_phase2(V, eps, X, p):
    print('in phase 2')
    (n, m) = np.shape(X)
    spending = np.zeros(n)
    for j in range(n):
        spending[j] = np.dot(p, X[j, :])
    jstar = np.argmin(spending) # least spender
    G = build_augmented_mbb_graph(V, X, p)
    H = build_hierarchy(jstar, G, X)
    l = 1
    if l in H:
        while len(H[l]) > 0 and False == is3epef1(X, p, eps):
            for k in H[l]:
                jstar_k_paths = nx.all_shortest_paths(G, source=('a', jstar), target=('a', k))
                for path in jstar_k_paths:
                    i = path[-2][1]
                    lminus1 = path[-3][1]
                    if spending[k] - p[i] > (1 + eps) * spending[jstar]:
                        X[k, i] = X[k, i] - 1
                        X[lminus1, i] = X[lminus1, i] + 1
                        return False, 2, X, p, jstar
            l = l + 1
            if not l in H:
                break
    if is3epef1(X, p, eps):
        return True, 0, X, p, jstar
    return False, 3, X, p, jstar


def market_phase3(V, eps, X, p, jstar):
    print('in phase 3')
    (n, m) = np.shape(V)

    # initialize things
    spending = np.zeros(n)
    for j in range(n):
        spending[j] = np.dot(p, X[j, :])
    BB = bpb(V, p)
    mbbr = dict()
    for j in range(n):
        mbbr[j] = np.max(BB[j, :])
    G = build_augmented_mbb_graph(V, X, p)
    H = build_hierarchy(jstar, G, X)
    Hagents = list()
    for l in range(n):
        for j in H[l]:
            Hagents.append(j)
    other_agents = [j for j in range(n) if j not in Hagents]
    Hitems = list()
    for j in Hagents:
        for i in range(m):
            if X[j, i] > 0:
                Hitems.append(i)
    other_items = [i for i in range(m) if i not in Hitems]

    # compute a1
    a1 = 0
    vals = list()
    for j in Hagents:
        for i in other_items:
            vals.append(mbbr[j] / (V[j, i] / p[i]))
    if len(vals) > 0:
        a1 = np.min(vals)

    # compute a2
    a2 = 0
    vals1 = list()
    for k in other_agents:
        vals2 = list()
        for i in range(m):
            if X[k, i] > 0:
                vals2.append(spending[k] - p[i])
        min_vals2 = np.min(vals2)
        vals1.append(min_vals2)
    if len(vals1) > 0 and spending[jstar] > 0:
        a2 = (1 / spending[jstar]) * np.max(vals1)
    if spending[jstar] == 0:
        a2 = np.infty

    # compute a3
    a3 = 0
    if len(other_agents) > 0 and spending[jstar] > 0:
        jhat = other_agents[np.argmin([spending[k] for k in other_agents])]
        s = 0
        while((1 + eps) ** s <= spending[jhat] / spending[jstar]):
            s = s + 1
        a3 = (1 + eps) ** s
    else:
        a3 = np.infty

    # compute a
    a = np.min([a1, a2, a3])

    for i in Hitems:
        p[i] = a * p[i]

    if a == a2:
        return True, 0, X, p
    return False, 2, X, p


def market_solve(V):
    (n, m) = np.shape(V)
    # Check if instance is Hall's condition violator
    V, eps = round_instance(V)
    status, X, p = market_phase1(V, eps)
    U = [np.dot(X[j, :], V[j, :]) for j in range(n)]
    P = [np.dot(X[j, :], p) for j in range(n)]
    if status:
        return status, X, p
    status = False
    next_phase = 2
    jstar = None # least spender
    while False == status:
        if next_phase == 2:
            status, next_phase, X, p, jstar = market_phase2(V, eps, X, p)
            U = [np.dot(X[j, :], V[j, :]) for j in range(n)]
            P = [np.dot(X[j, :], p) for j in range(n)]
        elif next_phase == 3:
            status, next_phase, X, p = market_phase3(V, eps, X, p, jstar)
        else:
            # raise Exception('Invalid next phase {}'.format(next_phase))
            return False, X, p
    return status, X, p
