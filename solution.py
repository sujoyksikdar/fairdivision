#####################################################################################
# ATTENTION! Remember to set B in config.py correctly
# mnw(V) # ILP to compute a Maximum Nash Welfare allocation for valuations V
# mnw_binary(V) # for binary valuations; compute a Maximum Nash Welfare allocation for valuations V
# leximin(V) # ILP to compute a leximin allocation for valuations V
# market(V) # EF1 version; compute an EF1 + PO allocation using market based algorithms for valuations V
#####################################################################################

import copy
import random
import logging
import numpy as np
import networkx as nx
from copy import deepcopy
from gurobipy import *

import config
from solution_util import *
from market import market_solve
from market_eq import market_eq_solve, market_eq_chores_solve


def mnw(V):
    Vval, valued = get_valued_instance(V)
    Vhalls, matched = get_halls_instance(Vval)
    status, w, U, Ahalls = mnw_solve(Vhalls)
    Ahat = recover_from_halls(Ahalls, Vval, matched)
    A = recover_from_valued(Ahat, V, valued)
    return status, w, U, A


def mnw_old(V):
    Vprime, matched = get_halls_instance(V)
    status, w, U, Aprime = mnw_solve(Vprime)
    A = recover_from_halls(Aprime, V, matched)
    return status, w, U, A


def mnw_solve(V):
    (n, m) = np.shape(V)

    model = Model('mnw')
    model.setParam('OutputFlag', 0)

    # add variables xvars for the allocation
    xvars = dict()
    for j in range(n):
        for i in range(m):
            xvars[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x_{}_{}'.format(j, i))

    # add variables for the welfare
    wvars = dict()
    for j in range(n):
        wvars[j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='w_{}'.format(j))

    # add allocation constraints
    for i in range(m):
        model.addConstr(1 == quicksum([xvars[j,i] for j in range(n)]))

    # add positive values constraints
    for j in range(n):
        model.addConstr(1 <= quicksum([V[j, i]*xvars[j, i] for i in range(m)]))

    # add upper bounds for welfare
    for j in range(n):
        for k in range(1,config.B):
            model.addConstr(wvars[j] <= np.log(k) + (np.log(k+1) - np.log(k)) * (quicksum([V[j, i] * xvars[j, i] for i in range(m)]) - k))

    # maximize Nash welfare
    model.setObjective(quicksum([wvars[j] for j in range(n)]), GRB.MAXIMIZE)

    model.optimize()

    status = model.Status

    A = np.zeros((n, m))
    for j in range(n):
        for i in range(m):
            A[j, i] = np.round(xvars[j, i].X) # rounded due to weird behavior where due to some numerical error,
            # 0.99... of item was allocated to an agent and was being set as not allocated.
    A = np.array(A).astype(int)
    w, U = nw(V, A)
    status = (status == GRB.OPTIMAL)
    return status, w, U, A


def test_mnw():
    V = np.array([[29, 41, 30], [1, 31, 68]])
    status, w, U, A = mnw(V)
    print(status, w, U, A)


def get_path(G, pair):
    (j, k) = pair
    return nx.shortest_path(G, source=j, target=k)


def get_reachability(V, G):
    (n, m) = np.shape(V)
    R = list()
    for j in range(n):
        for k in range(n):
            if nx.has_path(G, source=j, target=k):
                R.append((j, k))
    return R


def do_swaps(V, A, pair, G, labels):
    path = get_path(G, pair)
    B = deepcopy(A)
    cur = 0
    while cur < len(path) - 1:
        j = path[cur]
        k = path[cur + 1]
        i = labels[(j, k)]
        B[j, i] = 1
        B[k, i] = 0
        cur = cur + 1
    return B


def mnw_binary(V):
    (n, m) = np.shape(V)
    # initialize an allocation to start from
    A = max_match_allocation(V)
    Acur = deepcopy(A)
    for t in np.arange(1, 2*m*(n+1)*np.log(n*m) + 1, 1):
        Aprev = deepcopy(Acur)
        Gprev, labels = construct_alloc_graph(V, Aprev)
        R = get_reachability(V, Gprev)
        Apairs = list()
        nwApairs = list()
        for pair in R:
            Apair = do_swaps(V, Aprev, pair, Gprev, labels)
            Apairs.append(Apair)
            nwApair, U = nw(V, Apair)
            nwApairs.append(nwApair)
        max_nw_pairs = np.max(nwApairs)
        nw_prev, U = nw(V, Aprev)
        if max_nw_pairs > nw_prev:
            max_pair_idx = np.argmax(nwApairs)
            Acur = deepcopy(Apairs[max_pair_idx])
        else:
            return Aprev
    return Acur


def test_mnw_binary():
    V = np.array([[1, 0, 0], [0, 1, 1]])
    A = mnw_binary(V)
    nwA, U = nw(V, A)
    print('poly', nwA, A)
    status, w, U, A = mnw(V)
    nwA, U = nw(V, A)
    print('ilp', nwA, A)
    V = np.array([[1, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 1]])
    A = mnw_binary(V)
    nwA, U = nw(V, A)
    print('poly', nwA, A)
    status, w, U, A = mnw(V)
    nwA, U = nw(V, A)
    print('ilp', nwA, A)


# compute leximin allocation. works for goods only and chores only instances
# set chores=True for chores
def leximin(V, B=config.B, chores=False):
    (n, m) = np.shape(V)
    b = list()
    A = np.zeros((n, m))
    status = 0
    sw = 0
    U = np.zeros(n)
    for k in range(1, n+1):
        # print('iteration {}'.format(k))
        A = np.zeros((n, m))
        model = Model('leximin@{}'.format(k))
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 5 * 60)

        # add variable for lower bound for level k we are trying to find
        if k == 1:
            bvar = model.addVar(lb=-1*int(chores)*B, ub=(1-int(chores))*B, vtype=GRB.CONTINUOUS, name='b')
        else:
            bvar = model.addVar(lb=b[k-2], ub=(1-int(chores))*B, vtype=GRB.CONTINUOUS, name='b')

        # add variables for allocation
        xvars = dict()
        for j in range(n):
            for i in range(m):
                xvars[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x_{}_{}'.format(j, i))

        # add variables for utilities
        uvars = dict()
        for j in range(n):
            uvars[j] = model.addVar(lb=-1*int(chores)*B, ub=(1-int(chores))*B, vtype=GRB.CONTINUOUS, name='u_{}'.format(j))

        # add variables for lower bounds at each level
        yvars = dict() # for individual agents, indicate whether agent is above the lower bound for a level
        zvars = dict() # counting no. of agents
        for l in range(0, k):
            for j in range(n):
                yvars[j, l] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y_{}_{}'.format(j, l))
                zvars[l] = model.addVar(lb=0, ub=n, vtype=GRB.INTEGER, name='z_{}'.format(l))

        # add allocation constraints
        for i in range(m):
            model.addConstr(1 == quicksum([xvars[j,i] for j in range(n)]))

        # add constraints for utilities
        for j in range(n):
            model.addConstr(uvars[j] == quicksum([V[j,i]*xvars[j,i] for i in range(m)]))

        # add constraints for lower bounds
        for l in range(0, k - 1):
            bl = b[l]
            for j in range(n):
                model.addConstr(yvars[j, l] >= (1 + uvars[j] - bl)/B)
                model.addConstr(yvars[j, l] <= (B + uvars[j] - bl)/B)
            model.addConstr(zvars[l] == quicksum([yvars[j, l] for j in range(n)]))
            model.addConstr(zvars[l] >= n - l)

        for j in range(n):
            model.addConstr(yvars[j, k-1] >= (1 + uvars[j] - bvar)/B)
            model.addConstr(yvars[j, k-1] <= (B + uvars[j] - bvar)/B)
        model.addConstr(zvars[k-1] == quicksum([yvars[j, k-1] for j in range(n)]))
        model.addConstr(zvars[k-1] >= n - (k-1))

        model.setObjective(bvar, GRB.MAXIMIZE)

        model.optimize()

        status = model.Status
        # print('lower bound {}, at level {}, status {}, # meeting lower bound {}, lower bounds {}'.format(bvar.X, k-1, status, zvars[k-1].X, b))

        status = (status == GRB.OPTIMAL)
        if not status:
            return False, None, None, None

        b.append(bvar.X)
        # print('bvar: {}, lb {}, ub {}'.format(bvar.X, bvar.LB, bvar.UB))
        # print('uvars: {}'.format([(uvars[j].X, uvars[j].LB, uvars[j].UB) for j in range(n)]))
        # print('yvars: ', [[yvars[j, i].X for j in range(n)] for i in range(k)])
        # print('zvars: ', [zvars[i].X for i in range(k)])
        # print('uvars: ', [uvars[j].X for j in range(n)])

        for j in range(n):
            for i in range(m):
                A[j, i] = xvars[j, i].X
        A = np.array(A).astype(int)

        U = list()
        for j in range(n):
            u = np.dot(V[j, :], A[j, :])
            U.append(u)
        sw = np.sum(U)
        # print('iteration {}, bvar {}, utilities {}'.format(k, bvar.X, U))
    """
    print('---------------------------------------------')
    print('vals:\n', V)
    print('bounds ', b)
    print('utils: ', U)
    print('alloc:\n', A)
    print('---------------------------------------------')
    """
    return status, sw, U, A


def test_leximin():
    V = np.array([[100]])
    status, sw, U, A = leximin(V)
    print(V, status, sw, U, A)
    V = np.array([[29, 41, 30], [1, 31, 68]])
    status, sw, U, A = leximin(V)
    print(V, status, sw, U, A)
    V = np.array([[10, 60, 30], [50, 30, 20], [5, 60, 35]])
    status, sw, U, A = leximin(V)
    print(V, status, sw, U, A)


def market(V):
    print(V)
    Vval, valued = get_valued_instance(V)
    Vhalls, matched = get_halls_instance(Vval)
    print('Halls valuation\n{}'.format(Vhalls))
    status, Xhalls, p = market_solve(Vhalls)
    Xhat = recover_from_halls(Xhalls, Vval, matched)
    X = recover_from_valued(Xhat, V, valued)
    return status, X, p


def market_eq(V):
    Vval, valued = get_valued_instance(V)
    Vhalls, matched = get_halls_instance(Vval)
    # print('Halls valuation\n{}'.format(Vhalls))
    status, Xhalls, p = market_eq_solve(Vhalls)
    Xhat = recover_from_halls(Xhalls, Vval, matched)
    X = recover_from_valued(Xhat, V, valued)
    return status, X, p
