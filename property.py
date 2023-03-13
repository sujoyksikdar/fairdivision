###########################################################################
# Inputs: valuations V, allocation A
# Output: True/False
# is[eq/eq1/eqx/ef/ef1/efx/po](V, A)
# WARNING: ispo ILP is sensitive to numerical issues when "budget" is large
# IMPORTANT:
#     We will assume that goods are assigned a value >= 0 in V
#     and that chores are assigned a value <= 0 in V
# NOTES:
#     tested for goods only instances
#     not qualified for chores only or mixed goods and chores instances
#     for definitions for the case of goods only instances see Freeman et al. "Equitable Allocations of Indivisible Goods." In IJCAI 2019.
#     for definitions for the case of chores only instances see Freeman et al. "Equitable Allocations of Indivisible Chores." In AAMAS 2020.
# TODO:
#     add chores functionality
###########################################################################

import logging
import numpy as np
from gurobipy import *

import config


def iseq(V, A, chores=False):
    """
    Input:
        V: a valuation profile; an n x m 2D numpy array consisting of integers whose i.j-th entry V[i,j] represents agent i's the value for item j
        A: an allocation; a binary n x m 2D numpy array consisting of 0s and 1s
        chores: True / False indicating whether the input V represents a chores instance (True) or goods instance (False)
    Output:
        True / False: return True if A satisfies EQ w.r.t. the input valuation profile V
    """
    (n, m) = np.shape(V)
    # compute the utility experienced by each agent from allocation A according to valuations in V
    U = np.zeros(n) 
    for j in range(n):
        v = V[j,:]
        a = A[j,:]
        u = np.dot(v, a)
        U[j] = u
    # check for equitability (EQ)
    if all(u == U[0] for u in U):
        return True
    return False


def iseq1(V, A, chores=False):
    """
    Input:
        V: a valuation profile; an n x m 2D numpy array consisting of integers whose i.j-th entry V[i,j] represents agent i's the value for item j
        A: an allocation; a binary n x m 2D numpy array consisting of 0s and 1s
        chores: True / False indicating whether the input V represents a chores instance (True) or goods instance (False)
    Output:
        True / False: return True if A satisfies equitability-up-to-one-item (EQ1) w.r.t. the input valuation profile V
    """
    (n, m) = np.shape(V)
    for j1 in range(n):
        for j2 in range(n):
            if j1 == j2:
                continue
            v1 = V[j1,:]
            v2 = V[j2,:]
            a1 = A[j1,:]
            a2 = A[j2, :]
            u1 = np.dot(v1, a1)
            u2vec = np.multiply(v2, a2)
            # for chores: Check if 2's jealousy for 1 can be eliminated by removing a chore from 1
            if chores:
                if any(np.sum(u2vec[np.arange(m) != i]) >= u1 for i in range(m)):
                    continue
                else:
                    return False
            # for goods: Check if 1's jealousy for 2 can be eliminated by removing a good from 2
            if any(np.sum(u2vec[np.arange(m) != i]) <= u1 for i in range(m)):
                continue
            else:
                return False
    return True


def isdupeq1(V, A, chores=False):
    """
    Input:
        V: a valuation profile; an n x m 2D numpy array consisting of integers whose i.j-th entry V[i,j] represents agent i's the value for item j
        A: an allocation; a binary n x m 2D numpy array consisting of 0s and 1s
        chores: True / False indicating whether the input V represents a chores instance (True) or goods instance (False)
    Output:
        True / False: return True if A is equitabile-up-to-one-duplicated-item (DEQ1) w.r.t. the input valuation profile V
    """
    (n, m) = np.shape(V)
    for j1 in range(n):
        for j2 in range(n):
            if j1 == j2:
                continue
            v1 = V[j1,:]
            v2 = V[j2,:]
            a1 = A[j1,:]
            a2 = A[j2, :]
            u1 = np.dot(v1, a1)
            u2vec = np.multiply(v2, a2)
            # for chores: Check if 2's jealousy for 1 can be eliminated by removing a chore from 1
            if chores:
                if any(np.sum(u2vec[np.arange(m) != i]) >= u1 + u2vec[i] for i in range(m)):
                    continue
                else:
                    return False
            # for goods: Check if 1's jealousy for 2 can be eliminated by removing a good from 2
            if any(np.sum(u2vec[np.arange(m) != i]) <= u1 for i in range(m)):
                continue
            else:
                return False
    return True


def iseqx(V, A, chores=False):
    """
    Input:
        V: a valuation profile; an n x m 2D numpy array consisting of integers whose i.j-th entry V[i,j] represents agent i's the value for item j
        A: an allocation; a binary n x m 2D numpy array consisting of 0s and 1s
        chores: True / False indicating whether the input V represents a chores instance (True) or goods instance (False)
    Output:
        True / False: return True if A is envy-free-up-to-any-item (EQX) w.r.t. the input valuation profile V
    """
    (n, m) = np.shape(V)
    for j1 in range(n):
        for j2 in range(n):
            if j1 == j2:
                continue
            v1 = V[j1, :]
            v2 = V[j2, :]
            a1 = A[j1, :]
            a2 = A[j2, :]
            u1 = np.dot(v1, a1)
            u2vec = np.multiply(v2, a2)
            # for chores: Check if removing every negatively valued chore from u2 eliminates 2's jealousy
            if chores:
                u2neg = np.round(np.array([u2vec[i] for i in range(m) if u2vec[i] < 0]))
                if len(u2neg) <= 1:
                    continue
                if all(np.sum(u2neg[np.arange(len(u2neg)) != i]) >= u1 for i in range(len(u2neg))):
                    continue
                else:
                    condition = [np.sum(u2neg[np.arange(len(u2neg)) != i]) >= u1 for i in range(len(u2neg))]
                    violation = [np.sum(u2neg[np.arange(len(u2neg)) != i]) - u1 for i in range(len(u2neg))]
                    # print('agent {} jealous of {}, u2neg {}, u2vec {}, u2vec rounded {}, utils {}, {}, vals {}, {}, condition {}, violation {}'.format(j2, j1, u2neg, u2vec, np.round(u2vec), np.sum(u2vec), u1, v2, v1, condition, violation))
                    # sys.exit()
                    return False
            # for goods: Check if removing every positively valued good from u2 eliminates 1's jealousy
            u2pos = np.array([u2vec[i] for i in range(m) if u2vec[i] > 0])
            if len(u2pos) <= 1:
                continue
            if all(np.sum(u2pos[np.arange(len(u2pos)) != i]) <= u1 for i in range(len(u2pos))):
                continue
            else:
                return False
    return True


def isdupeqx(V, A, chores=False):
    """
    Input:
        V: a valuation profile; an n x m 2D numpy array consisting of integers whose i.j-th entry V[i,j] represents agent i's the value for item j
        A: an allocation; a binary n x m 2D numpy array consisting of 0s and 1s
        chores: True / False indicating whether the input V represents a chores instance (True) or goods instance (False)
    Output:
        True / False: return True if A is equitabile-up-to-any-duplicated-item (DEQX) w.r.t. the input valuation profile V
    """
    (n, m) = np.shape(V)
    for j1 in range(n):
        for j2 in range(n):
            if j1 == j2:
                continue
            v1 = V[j1, :]
            v2 = V[j2, :]
            a1 = A[j1, :]
            a2 = A[j2, :]
            u1 = np.dot(v1, a1)
            u2vec = np.multiply(v2, a2)
            # for chores: Check if removing every negatively valued chore from u2 eliminates 2's jealousy
            if chores:
                u2neg = np.round(np.array([u2vec[i] for i in range(m) if u2vec[i] < 0]))
                if len(u2neg) <= 1:
                    continue
                if all(np.sum(u2neg[np.arange(len(u2neg)) != i]) >= u1 + u2neg[i] for i in range(len(u2neg))):
                    continue
                else:
                    condition = [np.sum(u2neg[np.arange(len(u2neg)) != i]) >= u1 + u2neg[i] for i in range(len(u2neg))]
                    violation = [np.sum(u2neg[np.arange(len(u2neg)) != i]) - (u1 + u2neg[i]) for i in range(len(u2neg))]
                    print('agent {} jealous of {}, u2neg {}, u2vec {}, u2vec rounded {}, utils {}, {}, vals {}, {}, condition {}, violation {}'.format(j2, j1, u2neg, u2vec, np.round(u2vec), np.sum(u2vec), u1, v2, v1, condition, violation))
                    # sys.exit()
                    return False
            # for goods: Check if removing every positively valued good from u2 eliminates 1's jealousy
            u2pos = np.array([u2vec[i] for i in range(m) if u2vec[i] > 0])
            if len(u2pos) <= 1:
                continue
            if all(np.sum(u2pos[np.arange(len(u2pos)) != i]) <= u1 for i in range(len(u2pos))):
                continue
            else:
                return False
    return True


def isef(V, A, chores=False):
    """
    Input:
        V: a valuation profile; an n x m 2D numpy array consisting of integers whose i.j-th entry V[i,j] represents agent i's the value for item j
        A: an allocation; a binary n x m 2D numpy array consisting of 0s and 1s
        chores: True / False indicating whether the input V represents a chores instance (True) or goods instance (False)
    Output:
        True / False: return True if A is envy-free (EF) w.r.t. the input valuation profile V
    """
    (n, m) = np.shape(V)
    for j1 in range(n):
        for j2 in range(n):
            if j1 == j2:
                continue
            v1 = V[j1,:]
            a1 = A[j1,:]
            a2 = A[j2, :]
            u1 = np.dot(v1, a1)
            u12 = np.dot(v1, a2)
            if u12 > u1:
                return False
    return True


def isef1(V, A, chores=False):
    """
    Input:
        V: a valuation profile; an n x m 2D numpy array consisting of integers whose i.j-th entry V[i,j] represents agent i's the value for item j
        A: an allocation; a binary n x m 2D numpy array consisting of 0s and 1s
        chores: True / False indicating whether the input V represents a chores instance (True) or goods instance (False)
    Output:
        True / False: return True if A is envy-free-up-to-one-item (EF1) w.r.t. the input valuation profile V
    """
    (n, m) = np.shape(V)
    for j1 in range(n):
        for j2 in range(n):
            if j1 == j2:
                continue
            a1 = A[j1,:]
            a2 = A[j2, :]
            # for chores: Check if eliminating a chore from 2 eliminates envy towards 1
            if chores:
                v2 = V[j2,:]
                u21 = np.dot(v2, a1)
                u2vec = np.multiply(v2, a2)
                if any(np.sum(u2vec[np.arange(m) != i]) >= u21 for i in range(m)):
                    continue
                else:
                    return False
            # for goods: Check if eliminating a good from 2 eliminates 1's envy towards 2
            v1 = V[j1,:]
            u1 = np.dot(v1, a1)
            u12vec = np.multiply(v1, a2)
            if any(np.sum(u12vec[np.arange(m) != i]) <= u1 for i in range(m)):
                continue
            else:
                return False
    return True


def isefx(V, A, chores=False):
    """
    Input:
        V: a valuation profile; an n x m 2D numpy array consisting of integers whose i.j-th entry V[i,j] represents agent i's the value for item j
        A: an allocation; a binary n x m 2D numpy array consisting of 0s and 1s
        chores: True / False indicating whether the input V represents a chores instance (True) or goods instance (False)
    Output:
        True / False: return True if A is envy-free-up-to-any-item (EFX) w.r.t. the input valuation profile V
    """

    (n, m) = np.shape(V)
    for j1 in range(n):
        for j2 in range(n):
            if j1 == j2:
                continue
            a1 = A[j1, :]
            a2 = A[j2, :]
            # for chores: Check if eliminating every chore from 2 eliminates 2's envy towards 1
            if chores:
                v2 = V[j2,:]
                u2vec = np.multiply(v2, a2)
                u2neg = np.array([u2vec[i] for i in range(m) if u2vec[i] < 0])
                u21 = np.dot(v2, a1)
                if not len(u2neg) > 1:
                    continue
                if all(np.sum(u2neg[np.arange(len(u2neg)) != i]) >= u21 for i in range(len(u2neg))):
                    continue
                else:
                    return False
            # for goods: Check if eliminating every good from 2 eliminates 1's envy towards 2
            v1 = V[j1, :]
            u1 = np.dot(v1, a1)
            u12vec = np.multiply(v1, a2)
            u12pos = np.array([u12vec[i] for i in range(m) if u12vec[i] > 0])
            if not len(u12pos) > 1:
                continue
            if all(np.sum(u12pos[np.arange(len(u12pos)) != i]) <= u1 for i in range(len(u12pos))):
                continue
            else:
                return False
    return True


def ispo(V, A, chores=False, tol=10**-5):
    """
    Input:
        V: a valuation profile; an n x m 2D numpy array consisting of integers whose i.j-th entry V[i,j] represents agent i's the value for item j
        A: an allocation; a binary n x m 2D numpy array consisting of 0s and 1s
        chores: True / False indicating whether the input V represents a chores instance (True) or goods instance (False)
    Output:
        True / False: return True if A satisfies Pareto optimality (PO) w.r.t. the input valuation profile V
    Notes:
        This procedure works by solving an ILP to find an allocation B with maximum possible total utilitarian welfare 
        such that every agent has at least the same utility in B as they do in A,
        and then checking whether B Pareto dominates A by by checking if there is an agent who has strictly greater utility in B.
        Notice that the ILP should always have a feasible solution, since A is a feasible solution.

    """
    (n, m) = np.shape(V)
    assert(A.shape == (n,m))
    assert(np.all(np.sum(A, axis=0) == 1))
    U = np.zeros(n)
    for j in range(n):
        vj = V[j, :]
        aj = A[j, :]
        U[j] = np.dot(vj, aj)
    U = U.astype(int)
    Vmax = [np.sum(V[j, :]) for j in range(n)]

    model = Model('po')
    model.setParam('OutputFlag', 0)
    model.setParam('IntFeasTol', tol)

    # add variables xvars for the allocation
    xvars = dict()
    for j in range(n):
        for i in range(m):
            xvars[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x_{}_{}'.format(j, i))

    # add variables vbvars for agents' values for allocation B
    # notice that the lower and upper bounds on the value of vbvars are constraints ensuring that 
    # every agent j values its bundle in B at least as much as it values its bundle in A
    # in a goods instance:
    #   the lower bound is the agent's value (utility) for its bundle in A
    #   the upper bound is the value of all goods
    # in a chores instance:
    #   the lower bound is the agent's value (disutility) for its bundle in A
    #   the upper bound is 0, i.e., its value for receiving zero chores
    vbvars = dict()
    for j in range(n):
        vbvars[j] = model.addVar(lb=U[j], ub=int(1 - chores)*np.sum(V[j, :]), vtype=GRB.INTEGER, name='vb_{}'.format(j))
    print(j, U[j], int(chores)*np.sum(V[j, :]))

    # add allocation constraints
    for i in range(m):
        model.addConstr(1 == quicksum([xvars[j, i] for j in range(n)]))

    # add value constraints
    for j in range(n):
        model.addConstr(vbvars[j] == quicksum([V[j, i] * xvars[j, i] for i in range(m)]))

    # add Pareto dominance objective
    model.setObjective(quicksum([vbvars[j] for j in range(n)]), GRB.MAXIMIZE)

    model.optimize()
    status = model.Status
    print(status)

    if int(status) == int(GRB.INFEASIBLE):
        if tol < 0.1:
            model.reset()
            logging.debug('ILP infeasible, retry with tolerance {}'.format(tol * 10))
            return ispo2(V, A, tol=tol*10)
        logging.debug('ILP infeasible with tolerance {}'.format(tol))

    # print('status ', model.Status)
    B = np.zeros((n, m))
    for j in range(n):
        for i in range(m):
            B[j, i] = xvars[j, i].X
    UB = np.zeros(n)
    for j in range(n):
        UB[j] = np.dot(B[j, :], V[j, :])
    """ Diagnostics
    print('improved allocation:\n', B)
    print('utilities under improved allocation: ', UB)
    print('utilities under original allocation: ', U)
    print('improvements: ', UB - U)
    """

    if int(status) == int(GRB.OPTIMAL):
        if np.sum([vbvars[j].X for j in range(n)]) <= np.sum([U[j] for j in range(n)]):
            return True
        else:
            if np.any(UB - U > 0.1):
                return False
            else:
                logging.debug('more numerical issues, claimed improvements {}, actual improvements {}'.format(zvar.X, UB-U))
                return True
    return False


def check_prop(prop, V, A, chores=False):
    if prop == 'eq':
        return iseq(V, A, chores=chores)
    if prop == 'eq1':
        return iseq1(V, A, chores=chores)
    if prop == 'dupeq1':
        return isdupeq1(V, A, chores=chores)
    if prop == 'eqx':
        return iseqx(V, A, chores=chores)
    if prop == 'dupeqx':
        return isdupeqx(V, A, chores=chores)
    if prop == 'ef':
        return isef(V, A, chores=chores)
    if prop == 'ef1':
        return isef1(V, A, chores=chores)
    if prop == 'efx':
        return isefx(V, A, chores=chores)
    if prop == 'po':
        try:
            return ispo2(V, A, chores=chores)
        except:
            return ispo(V, A, chores=chores)
