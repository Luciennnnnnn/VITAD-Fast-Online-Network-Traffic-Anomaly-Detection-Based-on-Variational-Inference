#coding:utf-8
import time
import os
import heapq
import copy
import json
import sys
from math import pi, pow, sqrt, log, log10
import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.random import random_kruskal
from scipy.special import digamma, gamma
from scipy.stats import zscore
from imageio import imread, imsave
import scipy.signal
from numpy.linalg import norm, svd, inv, det
import matplotlib
matplotlib.use('Agg')
sys.path.append("../..")

from utils.utils import *


def BCPF_IC(Y, outliers_p, maxRank, maxiters, tol=1e-5, verbose=1, init='ml'):
    #Bayesian CP Factorization for Tensor Completion
    #
    #   [model] = BCPF_TC(Y, 'PARAM1', val1, 'PARAM2', val2, ...)
    #
    #   INPUTS
    #   Y - Input tensor
    #   'epsilon' - Binary(0 - 1) missing indicator tensor of same size as Y
    #          (0: missing; 1: observed)
    #   'init' - Initialization method
    #          - 'ml': SVD initilization(default)
    #          - 'rand': Random matrices of rank(larger than true rank)
    #   'dimRed' - True: Remove unnecessary components automaticly(default)
    #            - False: Not remove
    #   'maxiters' - max number of iterations(default: 100)
    #   'tol' - lower band change tolerance for convergence dection (default: 1e-5)
    #   'noise' - whether noise is updated
    #           - 'on': update noise parameter(default)
    #           - 'off': fixed noise parameter(1e-5)
    #   'predVar' - Predictive distribution
    #             - 1: compute and output
    #             - 0: doesnot compute(default)
    #   'verbose' - visualization of results
    #             - 0: no
    #             - 1: text(default)
    #             - 2: image plot
    #             - 3: hinton plot(very slow)
    #   OUTPUTS
    #       result - Model results
    #
    #   Example:
    #
    #       [model] = BCPF_TC(Y, outliers_p, 16, 20, 1e-5, 1, 'ml');
    #
    #   < Fast-Online-Network-Traffic-Anomaly-Detection-Based-on-Variational-Inference >
    #  Set parameters from input or by using defaults

    R = maxRank
    dimY = Y.shape
    N = Y.ndim
    T = dimY[2]
    outliers_count = np.sum(outliers_p, (0, 1))
    # Initialization
    a_tau0 = 1e-6
    b_tau0 = 1e-6

    tau = 1e4  # E[tau]
    dscale = 1

    Z0 = []
    ZSigma0 = []

    for n in range(N-1):
        if init == 'rand':
            Z0.append(np.random.randn(dimY[n], R))  # E[A ^ (n)]
        elif init == 'ml':
            U, S, _ = np.linalg.svd(unfold(Y, n), full_matrices=False)
            S = np.diag(S)
            Z0.append(np.dot(U[:, :R], np.square(S[:R, :R])))
        ZSigma0.append(np.zeros([R, R, dimY[n]]))  # covariance of A ^ (n)
        for i in range(dimY[n]):
            ZSigma0[n][:, :, i] = np.eye(R)

    # U, S, _ = np.linalg.svd(unfold(Y, 2), full_matrices=False)
    # S = np.diag(S)
    # z_tmp = np.dot(U[:, :R], np.square(S[:R, :R]))

    Z = []
    ZSigma = []
    for n in range(N - 1):
        Z.append(Z0[n])  # E[A ^ (n)]
        ZSigma.append(ZSigma0[n])

    # --------- E(aa') = cov(a,a) + E(a)E(a')----------------
    EZZT = []
    for n in range(N-1):
        EZZT.append(np.reshape(ZSigma0[n], [R*R, dimY[n]], 'F').T)#向量化的B^(n)
    #         EZZT{n} = (reshape(ZSigma{n}, [R*R, dimY(n)]))' + khatrirao_fast(Z{n}',Z{n}')'\

    Z0_t = np.random.randn(T, R)
    Z_t = np.random.randn(T, R)
    # elif init == 'ml':
    #     Z0_t = np.expand_dims(z_tmp[t, :], axis=0)
    #     Z_t = np.expand_dims(z_tmp[t, :], axis=0)

    ZSigma0_t = np.expand_dims(np.eye(R), 2)
    ZSigma0_t = ZSigma0_t.repeat(T, axis=2)
    ZSigma_t = np.zeros_like(ZSigma0_t)

    sigma_E0 = np.ones([dimY[0], dimY[1], T])
    E0 = np.zeros([dimY[0], dimY[1], T])
    # E0 = np.random.randn(dimY[0], dimY[1], 1)

    sigma_E = np.ones_like(sigma_E0)
    E = np.zeros_like(E0)

    RSE = []
    TPRS = []
    FPRS = []
    count = 0
    false_locations = []
    nObs = np.prod(dimY[0:2])
    O = np.ones([dimY[0], dimY[1], 1])
    for t in range(T):
        #  Model learning
        #print(t)
        LB = []
        #if init == 'rand':
        #Z0_t = np.random.randn(1, R)
        #Z_t = np.random.randn(1, R)
        # elif init == 'ml':
        #     Z0_t = np.expand_dims(z_tmp[t, :], axis=0)
        #     Z_t = np.expand_dims(z_tmp[t, :], axis=0)

        #ZSigma0_t = np.expand_dims(np.eye(R), 2)
        #ZSigma_t = np.eye(R)

        EZZT_t = np.reshape(ZSigma0_t[:, :, t], [R * R, 1], 'F').T

        #sigma_E0 = np.ones([dimY[0], dimY[1], 1])
        #E0 = np.zeros([dimY[0], dimY[1], 1])
        #E0 = np.random.randn(dimY[0], dimY[1], 1)

        #E = np.zeros_like(E0)
        #sigma_E = np.ones_like(sigma_E0)

        C = np.expand_dims(Y[:, :, t], 2)
        a_tauN = 1e-6
        b_tauN = 1e-6
        for it in range(maxiters):
            # Update factor matrices
            for n in range(N-1):
                # compute E(Z_{\n}^{T} Z_{\n})
                ENZZT = np.reshape(np.dot(khatri_rao([EZZT[0], EZZT[1], EZZT_t], skip_matrix=n, reverse=bool).T,
                                      unfold(O, n).T), [R, R, dimY[n]], 'F')
                # compute E(Z_{\n})
                FslashY = np.dot(khatri_rao([Z[0], Z[1], np.expand_dims(Z_t[t, :], axis=0)], skip_matrix=n, reverse=bool).T,
                                 unfold((C-np.expand_dims(E[:, :, t], axis=2)) * O, n).T)

                for i in range(dimY[n]):
                    ZSigma[n][:, :, i] = inv(tau * ENZZT[:, :, i] + inv(ZSigma0[n][:, :, i]))
                    Z[n][i, :] = np.squeeze((np.dot(ZSigma[n][:, :, i], (inv(ZSigma0[n][:, :, i]).dot(np.expand_dims(Z0[n][i, :], 1))
                                                    + tau*np.expand_dims(FslashY[:, i], 1)))).T)

                EZZT[n] = (np.reshape(ZSigma[n], [R * R, dimY[n]], 'F') + khatri_rao([Z[n].T, Z[n].T])).T

            ENZZT = np.reshape(np.dot(khatri_rao([EZZT[0], EZZT[1], EZZT_t], skip_matrix=2, reverse=bool).T, unfold(O, 2).T),
                               [R, R, 1], 'F')
            # compute E(Z_{\n})
            FslashY = np.dot(khatri_rao([Z[0], Z[1], Z_t[t, :]], skip_matrix=2, reverse=bool).T,
                             unfold((C-E[:, :, t]) * O, 2).T)

            # parameter in current sample point
            ZSigma_t[:, :, t] = inv(tau * ENZZT[:, :, 0] + inv(ZSigma0_t[:, :, t]))

            Z_t[t, :] = (np.dot(ZSigma_t[:, :, t], (inv(ZSigma0_t[:, :, t]).dot(np.reshape(Z0_t[t, :], [R, 1])) +
                                                                 tau*np.expand_dims(FslashY[:, 0], 1)))).T

            EZZT_t = (np.reshape(np.reshape(Z_t[t, :], [R, 1]).dot(np.expand_dims(Z_t[t, :], axis=0)), [R*R, 1], 'F') + \
                             np.reshape(ZSigma_t[:, :, t], [R*R, 1], 'F')).T

            # Update latent tensor X
            X = np.squeeze(tl.kruskal_tensor.kruskal_to_tensor([Z[0], Z[1], np.expand_dims(Z_t[t, :], axis=0)]))   #

            #  The most time and space consuming part
            EX2 = np.dot(np.dot(tensor_to_vec(O).T, khatri_rao([EZZT[0], EZZT[1],
                                EZZT_t], reverse=bool)), np.ones([R * R, 1]))

            EE2 = np.sum(np.square(E[:, :, t]) + sigma_E[:, :, t])
            err = np.dot(tensor_to_vec(C).T, tensor_to_vec(C)) - 2 * tensor_to_vec(C).T \
                .dot(tensor_to_vec(X)) -2*np.dot(tensor_to_vec(C).T, tensor_to_vec(E[:, :, t])) + \
                  2*np.dot(tensor_to_vec(X).T, tensor_to_vec(E[:, :, t])) + EX2 + EE2

            #  update noise tau
            a_tauN = (a_tau0 + 0.5 * nObs)  # a_MnObs
            b_tauN = (b_tau0 + 0.5 * err)  # b_M

            tau = a_tauN / b_tauN

            #Update sparse matrix E
            sigma_E[:, :, t] = np.reciprocal(tau + np.reciprocal(sigma_E0[:, :, t]))
            E[:, :, t] = (sigma_E[:, :, t] * (E0[:, :, t]/sigma_E0[:, :, t] + tau * np.squeeze(C-np.expand_dims(X, axis=2))))

            temp1 = -0.5 * nObs * safelog(2 * pi) + 0.5 * nObs * (digamma(a_tauN) - safelog(b_tauN)) \
                    - 0.5 * tau * err
            temp2 = 0
            for n in range(N-1):
                for i in range(dimY[n]):
                  temp2 += -0.5 * R * safelog(2 * pi) + 0.5 * safelog(det(ZSigma0[n][:, :, i])) - 0.5 * \
                           np.trace(ZSigma0[n][:, :, i].dot(ZSigma[n][:, :, i])) - 0.5 * \
                           np.dot(np.expand_dims(Z[n][i, :], 0), ZSigma0[n][:, :, i]).dot(np.expand_dims(Z[n][i, :], 1))

            temp2 += -0.5 * R * safelog(2 * pi) + 0.5 * safelog(det(ZSigma0_t[:, :, t])) - 0.5 * \
                           np.trace(ZSigma0_t[:, :, t].dot(ZSigma_t[:, :, t])) - 0.5 * \
                           np.squeeze(np.dot(np.expand_dims(Z_t[t, :], axis=0), ZSigma0_t[:, :, t]).dot(np.reshape(Z_t[t, :], [R, 1])))

            temp3 = -safelog(gamma(a_tau0)) + a_tau0 * safelog(b_tau0) + (a_tau0 - 1) * (
                        digamma(a_tauN) - safelog(b_tauN)) \
                    - b_tau0 * (a_tauN / b_tauN)

            temp4 = 0
            for n in range(N-1):
                for i in range(ZSigma[n].shape[2]):
                    temp4 += 0.5 * safelog(np.linalg.det(ZSigma[n][:, :, i])) + 0.5 * R * (1 + safelog(2 * pi))

            temp4 += 0.5 * safelog(np.linalg.det(ZSigma_t[:, :, t])) + 0.5 * R * (1 + safelog(2 * pi))

            temp5 = safelog(gamma(a_tauN)) - (a_tauN - 1) * digamma(a_tauN) - safelog(b_tauN) + a_tauN

            temp6 = -0.5 * nObs * safelog(2 * pi) - np.sum(0.5 * safelog(sigma_E0[:, :, t]) - 0.5*(np.square(E[:, :, t]) + sigma_E[:, :, t] - 2*E[:, :, t]*E[:, :, t] + np.square(E0[:, :, t]))/sigma_E0[:, :, t])

            temp7 = 0.5 * nObs * safelog(2 * pi) + np.sum(0.5 * safelog(sigma_E[:, :, t]) + 0.5*(np.square(E[:, :, t]) + sigma_E[:, :, t] - 2*E[:, :, t]*E0[:, :, t] + np.square(E[:, :, t]))/sigma_E[:, :, t])
            LB.append(temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7)

            # Display progress
            if it > 2:
                LBRelChan = abs(LB[it] - 2*LB[it-1] + LB[it-2])/-LB[1]
            else:
                LBRelChan = 0
            #
            # if verbose:
            #     print('Iter. %d: RelChan = %g' % (it, LBRelChan))
            # Convergence check
            if it > 5 and (abs(LBRelChan) < tol):
                #print('======= Converged===========')
                break
        [TPR, FPR] = check(E[:, :, t], outliers_p[:, :, t], outliers_count[t], dimY)
        #false_locations.append(locations)
        count += TPR
        #X = np.expand_dims(X, 2)
        #RSE.append(norm(C-X-E) / norm(C))
        TPRS.append(TPR)
        FPRS.append(FPR)
        Z0 = copy.deepcopy(Z)
        ZSigma0 = copy.deepcopy(ZSigma)
        a_tau0 = a_tauN
        b_tau0 = b_tauN

    # Prepare the results
    #Z.append(Z_ST)
    #X = tl.kruskal_to_tensor(Z) * dscale

    model = {}
    # Output
    #model['X'] = X
    #model['RSE'] = RSE
    model['TPRS'] = TPRS
    model['FPRS'] = FPRS
    model['precision'] = count / T
    model['FPR'] = np.sum(FPRS) / T
    #model['ZSigma'] = ZSigma
    #model['false_locations'] = false_locations
    return model