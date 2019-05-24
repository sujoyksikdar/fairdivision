import pickle
import random
import numpy as np
import scipy.io

import config


def flip_biased_coin(p=0.5):
    return np.random.random() < p


def generate_binary_instance_withbias(m, n, p=0.5):
    V = np.zeros((n, m))
    for j in range(n):
        for i in range(m):
            V[j, i] = flip_biased_coin(p)
    return V


def generate_binary_instances_withbias(m, n, p, size, offset=0, loc=''):
    I = list()
    for i in range(size):
        V = generate_binary_instance_withbias(m, n, p=p)
        with open('{}{}_{}_{}.b{}instance'.format(loc, m, n, i+offset, p), 'wb') as fo:
            pickle.dump(V, fo)
        I.append(V)
    return I


def load_binary_instances_withbias(m, n, p, start=0, end=0, loc=''):
    I = list()
    for i in range(start, end+1):
        with open('{}{}_{}_{}.b{}instance'.format(loc, m, n, i, p), 'rb') as f:
            V = pickle.load(f)
            I.append(V)
    return I


def generate_positive_dirichlet_instance(m, n, B=config.B):
    V = list()
    for j in range(n):
        vals = np.random.dirichlet([10 for i in range(m)])
        vals = vals + 0.001
        vals = [np.ceil(B*v) for v in vals]
        while np.sum(vals) > B:
            i = np.random.randint(low=0, high=n)
            if vals[i] > 1:
                vals[i] = vals[i] - 1
        V.append(vals)
    V = np.array(V)
    return V


def generate_positive_dirichlet_instances(m, n, size):
    I = list()
    for i in range(size):
        V = generate_positive_dirichlet_instance(m, n)
        with open('{}_{}_{}.pdinstance'.format(m, n, i), 'wb') as fo:
            pickle.dump(V, fo)
        I.append(V)
    return I


def load_positive_dirichlet_instances(m, n, start=0, end=1, loc='data/'):
    I = list()
    for i in range(start, end+1):
        with open('{}{}_{}_{}.pdinstance'.format(loc, m, n, i), 'rb') as f:
            V = pickle.load(f)
            I.append(V)
    return I


def load_real_instances(filename='spliddit_goods_data.mat'):
    data = scipy.io.loadmat(filename)
    instances = data['valuations'][0]
    num_instances = np.shape(instances)[0]
    I = list()
    for k in range(num_instances):
        instance = instances[k]
        (n, m) = np.shape(instance)
        V = list()
        for j in range(n):
            v = list()
            for i in range(m):
                v.append(instance[j, i])
            V.append(v)
        V = np.array(V)
        I.append(V)
    return I
