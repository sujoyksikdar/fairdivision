import logging

from market import *


def round_eq_instance(V):
    (n, m) = np.shape(V)
    vmax = np.max(V)
    eps = 1/(16 * m * np.power(np.float64(vmax), 4)) # cast as float64 because for some reason python seems to use 32 bits internally somewhere
    Vprime = np.zeros((n, m))
    for j in range(n):
        for i in range(m):
            if V[j, i] > 0:
                Vprime[j, i] = (1 + eps) ** np.ceil(np.log(V[j, i]) / np.log(1 + eps))
    return Vprime, eps


def isepeq1(V, X, eps):
    (n, m) = np.shape(X)
    U = np.zeros(n)
    for j in range(n):
        U[j] = np.dot(V[j, :], X[j, :])
    for j in range(n):
        if np.sum(X[j, :]) <= 1:
            continue
        vals = [U[j] - V[j, i] for i in range(m) if X[j, i] > 0]
        for k in range(n):
            if np.all(vals > (1 + eps) * U[k]):
                return False
    return True


def market_eq_phase1(V, eps):
    (n, m) = np.shape(V)
    X = np.zeros((n, m))
    p = np.zeros(m)
    for i in range(m):
        ivals = V[:, i]
        j = np.argmax(ivals)
        X[j, i] = 1
        p[i] = V[j, i]
    if isepeq1(V, X, eps):
        return True, X, p
    return False, X, p


def market_eq_phase2(V, eps, X, p):
    (n, m) = np.shape(V)
    U = np.zeros(n)
    for j in range(n):
        U[j] = np.dot(V[j, :], X[j, :])
    jstar = np.argmin(U)
    G = build_augmented_mbb_graph(V, X, p)
    H = build_hierarchy(jstar, G, X)
    l = 1
    while len(H[l]) > 0 and False == isepeq1(V, X, eps):
        for k in H[l]:
            jstar_k_paths = nx.all_shortest_paths(G, source=('a', jstar), target=('a', k))
            for path in jstar_k_paths:
                i = path[-2][1]
                lminus1 = path[-3][1]
                if U[k] - V[k, i] > (1 + eps) * U[jstar]:
                    X[k, i] = X[k, i] - 1
                    X[lminus1, i] = X[lminus1, i] + 1
                    return False, 2, X, p, jstar
        l = l + 1
        if not l in H:
            break
    if isepeq1(V, X, eps):
        return True, 0, X, p, jstar
    return False, 3, X, p, jstar


def market_eq_phase3(V, eps, X, p, jstar):
    (n, m) = np.shape(V)

    # initialize things
    U = np.zeros(n)
    for j in range(n):
        U[j] = np.dot(V[j, :], X[j, :])
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

    # compute stuff
    vals = list()
    for j in Hagents:
        for i in other_items:
            vals.append(mbbr[j] / (V[j, i] / p[i]))
    D = np.min(vals)

    for i in Hitems:
        p[i] = p[i] * D
    return False, 2, X, p


def market_eq_solve(V):
    V, eps = round_eq_instance(V)
    status, X, p = market_eq_phase1(V, eps)
    if status:
        return status, X, p
    status = False
    next_phase = 2
    jstar = None # least spender
    while False == status:
        if next_phase == 2:
            status, next_phase, X, p, jstar = market_eq_phase2(V, eps, X, p)
        elif next_phase == 3:
            status, next_phase, X, p = market_eq_phase3(V, eps, X, p, jstar)
        else:
            return False, X, p
    return status, X, p
