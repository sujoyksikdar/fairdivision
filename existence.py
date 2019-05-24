##############################################################################
# exists_props(V, props, B=config.B, chores=False)
# props is a '+' delimited string of props.
# props can contain only one of eq/eq1/eqx, only one of ef/ef1/efx, and po
##############################################################################

import numpy as np
from gurobipy import *

import config


def exists_props(V, props, B=config.B, chores=False):
    if chores:
        return exists_props_chores(V, props, B)
    else:
        return exists_props_goods(V, props, B)


def exists_props_goods(V, props, B=config.B):
    (n, m) = np.shape(V)

    model = Model('exists_goods')
    model.setParam('OutputFlag', 0)

    # add variables xvars for the allocation
    xvars = dict()
    for j in range(n):
        for i in range(m):
            xvars[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x_{}_{}'.format(j, i))

    # add variables for agents' utilities
    uvars = dict()
    for j in range(n):
        uvars[j] = model.addVar(lb=0, ub=np.sum(V[j, :]), vtype=GRB.CONTINUOUS, name='u_{}'.format(j))

    # add variables for agents' utilities for other agents' allocations
    ujkvars = dict() # j's utility for k's bundle
    ujklvars = dict() # j's utility for k's bundle after removing item l if l is assigned. otherwise, 0 is subtracted from k's bundle
    for j in range(n):
        for k in range(n):
            ujkvars[j, k] = model.addVar(lb=0, ub=np.sum(V[j, :]), vtype=GRB.CONTINUOUS, name='ujk_{}_{}'.format(j, k))
            for l in range(m):
                ujklvars[j, k, l] = model.addVar(lb=0, ub=np.sum(V[j, :]), vtype=GRB.CONTINUOUS, name='ujkl_{}_{}_{}'.format(j, k, l))

    # add variables to check for envyfreeness
    efjkvars = dict()  # # items from k's bundle which when removed eliminate envy
    efjklvars = dict()  # indicator of whether removing l from k's bundle eliminates envy. see description of ujklvars
    if 'ef' in props or 'ef1' in props or 'efx' in props:
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                efjkvars[j, k] = model.addVar(lb=0, ub=m, vtype=GRB.CONTINUOUS, name='efjk_{}_{}'.format(j, k))
                for l in range(m):
                    efjklvars[j, k, l] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='efjkl_{}_{}_{}'.format(j, k, l))

    # add variables to check for equitability
    eqjkvars = dict()  # # items from k's bundle which restore equitability
    eqjklvars = dict()  # indicator of whether removing l from k's bundle restores equitability. see description of ujklvars
    if 'eq' in props or 'eq1' in props or 'eqx' in props:
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                eqjkvars[j, k] = model.addVar(lb=0, ub=m, vtype=GRB.CONTINUOUS, name='eqjk_{}_{}'.format(j, k))
                for l in range(m):
                    eqjklvars[j, k, l] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='eqjkl_{}_{}_{}'.format(j, k, l))

    # add variables for checking Pareto optimality
    bvars = dict() # for the Pareto improving allocation B
    vbvars = dict() # for agents' values for Pareto improving allocation B
    yvars = dict() # to check for strict improvements over X
    zvar = None # to count number of strict improvements
    if 'po' in props:
        # add variables bvars for the Pareto improving allocation B
        for j in range(n):
            for i in range(m):
                bvars[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='b_{}_{}'.format(j, i))

        # add variables vbvars for agents' values for Pareto improving allocation B
        for j in range(n):
            vbvars[j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='vb_{}'.format(j))

        # add variables yvars to check for strict improvements over X
        for j in range(n):
            yvars[j] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y_{}'.format(j))

        # add variable zvar to count number of strict improvements
        zvar = model.addVar(lb=0, ub=n, vtype=GRB.CONTINUOUS, name='z')

    # add constraints for allocation
    for i in range(m):
        model.addConstr(1 == quicksum([xvars[j, i] for j in range(n)]))

    # add constraints for utilities
    for j in range(n):
        model.addConstr(uvars[j] == quicksum([V[j, i]*xvars[j, i] for i in range(m)]))

    # add constraints for utility of j for k's bundle
    # add constraints for utility of j for k's bundle after removal of item l if assigned to k
    for j in range(n):
        for k in range(n):
            if j==k:
                continue
            model.addConstr(ujkvars[j, k] == quicksum([V[j, l]*xvars[k, l] for l in range(m)]))
            for l in range(m):
                model.addConstr(ujklvars[j, k, l] == ujkvars[j, k] - V[j, l]*xvars[k, l])

    # add constraints for envy-freeness
    if 'ef' in props or 'ef1' in props or 'efx' in props:
        # add constraints for j envying k
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                if 'ef' in props:
                    model.addConstr(uvars[j] >= ujkvars[j, k])
                else:
                    for l in range(m):
                        model.addConstr(efjklvars[j, k, l] >= (1 + uvars[j] - ujklvars[j, k, l]) / B)
                        model.addConstr(efjklvars[j, k, l] <= (B + uvars[j] - ujklvars[j, k, l]) / B)
                    model.addConstr(efjkvars[j, k] == quicksum([efjklvars[j, k, l] for l in range(m)]))
                    if 'ef1' in props:
                        model.addConstr(efjkvars[j, k] >= 1)
                    if 'efx' in props:
                        model.addConstr(efjkvars[j, k] >= quicksum([xvars[k, l] for l in range(m)]))

    # add constraints for equitability
    if 'eq' in props or 'eq1' in props or 'eqx' in props:
        # add constraints for j being inequitable w.r.t. k
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                if 'eq' in props:
                    model.addConstr(uvars[j] >= uvars[k])
                else:
                    for l in range(m):
                        model.addConstr(eqjklvars[j, k, l] >= (1 + uvars[j] - ujklvars[k, k, l]) / B)
                        model.addConstr(eqjklvars[j, k, l] <= (B + uvars[j] - ujklvars[k, k, l]) / B)
                    model.addConstr(eqjkvars[j, k] == quicksum([eqjklvars[j,k,l] for l in range(m)]))
                    if 'eq1' in props:
                        model.addConstr(eqjkvars[j, k] >= 1)
                    if 'eqx' in props:
                        model.addConstr(eqjkvars[j, k] >= quicksum([xvars[k, l] for l in range(m)]))

    # add constraints for Pareto optimality
    if 'po' in props:
        # add allocation constraints for Pareto improvement B
        for i in range(m):
            model.addConstr(1 == quicksum([bvars[j, i] for j in range(n)]))

        # add value constraints
        for j in range(n):
            model.addConstr(vbvars[j] == quicksum([V[j, i] * bvars[j, i] for i in range(m)]))

        # add constraints for weak improvement
        for j in range(n):
            model.addConstr(vbvars[j] >= uvars[j])

        # add strict improvement constraints
        for j in range(n):
            model.addConstr(yvars[j] >= (vbvars[j] - uvars[j]) / np.sum(V[j, :]))
            model.addConstr(yvars[j] <= vbvars[j] - uvars[j])
        model.addConstr(zvar == quicksum([yvars[j] for j in range(n)]))

        model.setObjective(zvar, GRB.MAXIMIZE)

    model.optimize()

    status = model.Status

    A = None
    if int(status) == int(GRB.OPTIMAL):
        A = np.zeros((n, m))
        for j in range(n):
            for i in range(m):
                A[j, i] = xvars[j, i].X
        if 'po' in props:
            if zvar.X == 0:
                return True, A
            return False, A
        return True, A
    return False, A
