import copy
import json
import os
import heapq
from math import pi, pow, sqrt, log, log10
import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.random import random_kruskal
from scipy.special import digamma, gamma
from scipy.stats import zscore
import scipy.signal
from numpy.linalg import norm, svd, inv, det
from scipy.stats import levy_stable


def safelog(x):
    if isinstance(x, np.ndarray):
        x[np.where(x < 1e-300)] = 1e-200
        x[np.where(x > 1e300)] = 1e300
        return np.log(x)
    else:
        if x < 1e-300:
            x = 1e-200
        if x > 1e300:
            x = 1e300
        return log(x)


def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), 'F')


def tensor_to_vec(tensor):
    return np.reshape(tensor, (-1, 1), 'F')


def hardmard(tensors):
    ans = np.ones_like(tensors[0])
    for i in range(len(tensors)):
        ans = ans * tensors[i]
    return ans


def choice(DIM, p):
    indices = np.random.choice(np.prod(DIM), int(round(np.prod(DIM) * p)), replace=False)
    X, Y, Z = [], [], []
    for i in range(indices.size):
        Z.append(indices[i] // (DIM[0] * DIM[1]))
        indices[i] %= DIM[0] * DIM[1]
        Y.append(indices[i] // DIM[0])
        X.append(indices[i] % DIM[0])
    return X, Y, Z


def generator(dataset_name, fraction, mu, sigma, SNR=None, distribution="Gaussian"):
    #print(os.getcwd())
    a = np.load(os.path.join(os.path.join('../../data', dataset_name), r'normlized_tensor.npy'))
    b = np.zeros_like(a, dtype=bool)
    DIM = a.shape

    #Add noise
    if SNR != None:
        sigma2 = np.var(tensor_to_vec(a))*(1 / (10 ** (SNR / 10)))
        GN = np.sqrt(sigma2) * np.random.randn(DIM[0], DIM[1], DIM[2])
        a = a + GN

    #Add outliers
    if distribution == "Gaussian":
        outliers = np.random.randn(DIM[0], DIM[1], DIM[2]) * sqrt(sigma) + mu
    elif distribution == "levy_stable":
        outliers = levy_stable.rvs(sigma, mu, size=(DIM[0], DIM[1], DIM[2]))

    locations = list(range(np.prod(DIM)))
    if fraction != 0:
        sampled_locations = np.random.choice(locations, int(len(locations) * fraction), replace=False)
        # print(len(sampled_locations))
        for x in sampled_locations:
            k = x // (DIM[0] * DIM[1])
            x %= (DIM[0] * DIM[1])
            i = x // DIM[1]
            j = x % DIM[1]
            b[i, j, k] = 1
            a[i, j, k] += outliers[i, j, k]
    return a, b


def generator2(dataset_name, fraction, mu, sigma, SNR=None, distribution="Gaussian"):
    #print(os.getcwd())
    a = np.load(os.path.join(os.path.join('../../data', dataset_name), r'normlized_tensor.npy'))
    b = np.zeros_like(a, dtype=bool)
    DIM = a.shape
    sigmas = [0.01, 0.05, 0.1, 0.5, 1]
    #Add noise
    if SNR != None:
        sigma2 = np.var(tensor_to_vec(a))*(1 / (10 ** (SNR / 10)))
        GN = np.sqrt(sigma2) * np.random.randn(DIM[0], DIM[1], DIM[2])
        a = a + GN

    for it in range(5):
        #Add outliers
        if distribution == "Gaussian":
            outliers = np.random.randn(DIM[0], DIM[1], DIM[2]) * sqrt(sigmas[it]) + mu
        elif distribution == "levy_stable":
            outliers = levy_stable.rvs(sigma, mu, size=(DIM[0], DIM[1], DIM[2]))

        locations = list(range(np.prod(DIM)))
        if fraction != 0:
            sampled_locations = np.random.choice(locations, int(len(locations) * fraction), replace=False)
            # print(len(sampled_locations))
            for x in sampled_locations:
                k = x // (DIM[0] * DIM[1])
                x %= (DIM[0] * DIM[1])
                i = x // DIM[1]
                j = x % DIM[1]
                b[i, j, k] = 1
                a[i, j, k] += outliers[i, j, k]
    return a, b


def topk(X, _k, dimY):
    a = []
    for i in range(dimY[0]):
        for j in range(dimY[1]):
            a.append((X[i, j], (i, j)))
    mx = heapq.nlargest(_k, a, key=lambda s: abs(s[0]))
    return mx


def check(e, outliers_p, epsilon, dimY):
    TP = 0
    FP = 0
    p = topk(e, epsilon, dimY)
    #false_locations = []
    for elem in p:
        s = elem[1]
        if outliers_p[s[0], s[1]]:
            TP += 1
        else:
            FP += 1
            #false_locations.append((s[0], s[1]))
    if epsilon == 0:
        TPR = 1
    else:
        TPR = TP / epsilon
    if epsilon == dimY[0] * dimY[1]:
        FPR = 0
    else:
        FPR = FP / (dimY[0] * dimY[1] - epsilon)
    return TPR, FPR#, false_locations