#coding:utf-8  将异常部分变为条件
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

from utils import *


def update(Y, O, params, N, maxRank, maxiters, tol=1e-5, verbose=0):
    Z0 = params['Z']
    ZSigma0 = params['ZSigma']
    coefficient = params['coefficient']
    EZZT = params['EZZT']
    EZZT_t_pre = params['EZZT_t_pre']
    Z_pre = params['Z_pre']
    O_pre = params['O_pre']
    a_tau0 = params['a_tau0']
    b_tau0 = params['b_tau0']
    tau = params['tau']
    L_pre = params['L_pre']

    #  Model learning
    R = maxRank
    nObs = np.sum(O)
    LB = []

    dimY = Y.shape
    C = np.expand_dims(Y, 2)

    Z = []
    ZSigma = []
    for n in range(N - 1):
        Z.append(np.random.randn(dimY[n], R))  # E[A ^ (n)]
        ZSigma.append(np.zeros([R, R, dimY[n]]))  # covariance of A ^ (n)
        for i in range(dimY[n]):
            ZSigma[n][:, :, i] = np.eye(R)

    #if init == 'rand':
    Z0_t = np.random.randn(1, R)
    Z_t = np.random.randn(1, R)
    # elif init == 'ml':
    #     Z0_t = np.expand_dims(z_tmp[t, :], axis=0)
    #     Z_t = np.expand_dims(z_tmp[t, :], axis=0)

    ZSigma0_t = np.expand_dims(np.eye(R), 2)
    ZSigma_t = np.eye(R)

    EZZT_t = np.reshape(ZSigma0_t, [R * R, 1], 'F').T

    sigma_E0 = np.ones([dimY[0], dimY[1], 1])
    E0 = np.zeros([dimY[0], dimY[1], 1])

    E = np.zeros_like(E0)
    sigma_E = np.ones_like(sigma_E0)

    a_tauN = 1e-6
    b_tauN = 1e-6

    for it in range(maxiters):
        for n in range(N-1):
            ENZZT = np.reshape(np.dot(khatri_rao([EZZT[0], EZZT[1], EZZT_t], skip_matrix=n, reverse=bool).T,
                                      unfold(O, n).T), [R, R, dimY[n]], 'F')
            ENZZT_prev = []
            for i in range(len(coefficient)):
                tmp = np.reshape(np.dot(khatri_rao([EZZT[0], EZZT[1], EZZT_t_pre[i]], skip_matrix=n, reverse=bool).T,
                                      unfold(O, n).T), [R, R, dimY[n]], 'F')
                ENZZT_prev.append(tmp)

            FslashY = np.dot(khatri_rao([Z[0], Z[1], Z_t], skip_matrix=n, reverse=bool).T,
                             unfold((C - E) * O, n).T)

            for i in range(len(coefficient)):
                FslashY += coefficient[i] * np.dot(khatri_rao([Z[0], Z[1], Z_pre[i]], skip_matrix=n, reverse=bool).T,
                             unfold(L_pre[i] * O_pre[i], n).T)

            for i in range(dimY[n]):
                ENZZT_sum = ENZZT[:, :, i]
                for j in range(len(coefficient)):
                    ENZZT_sum += coefficient[j] * ENZZT_prev[j][:, :, i]
                ZSigma[n][:, :, i] = inv(tau * ENZZT_sum + inv(ZSigma0[n][:, :, i]))
                Z[n][i, :] = np.squeeze(
                    (np.dot(ZSigma[n][:, :, i], (inv(ZSigma0[n][:, :, i]).dot(np.expand_dims(Z0[n][i, :], 1))
                                                 + tau * np.expand_dims(FslashY[:, i], 1)))).T)

            EZZT[n] = (np.reshape(ZSigma[n], [R * R, dimY[n]], 'F') + khatri_rao([Z[n].T, Z[n].T])).T

            ENZZT = np.reshape(np.dot(khatri_rao([EZZT[0], EZZT[1], EZZT_t], skip_matrix=2, reverse=bool).T, unfold(O, 2).T),
                               [R, R, 1], 'F')
            # compute E(Z_{\n})
            FslashY = np.dot(khatri_rao([Z[0], Z[1], Z_t], skip_matrix=2, reverse=bool).T,
                             unfold((C-E) * O, 2).T)

            ZSigma_t = inv(tau * ENZZT[:, :, 0] + inv(ZSigma0_t[:, :, 0]))

            Z_t = (np.dot(ZSigma_t, (inv(ZSigma0_t[:, :, 0]).dot(np.reshape(Z0_t, [R, 1])) +
                                                                 tau*np.expand_dims(FslashY[:, 0], 1)))).T

            EZZT_t = (np.reshape(np.reshape(Z_t, [R, 1]).dot(Z_t), [R*R, 1], 'F') + \
                             np.reshape(ZSigma_t, [R*R, 1], 'F')).T

            # Update latent tensor X
            X = np.squeeze(tl.kruskal_tensor.kruskal_to_tensor([Z[0], Z[1], Z_t]))

            #  The most time and space consuming part
            EX2 = np.dot(np.dot(tensor_to_vec(O).T, khatri_rao([EZZT[0], EZZT[1],
                                                                EZZT_t], reverse=bool)), np.ones([R * R, 1]))

            EE2 = np.sum(np.square(E) + sigma_E)
            err = np.dot(tensor_to_vec(C).T, tensor_to_vec(C)) - 2 * tensor_to_vec(C).T \
                .dot(tensor_to_vec(X)) - 2 * np.dot(tensor_to_vec(C).T, tensor_to_vec(E)) + \
                  2 * np.dot(tensor_to_vec(X).T, tensor_to_vec(E)) + EX2 + EE2
            nObs_sum = nObs
            errs = []
            for i in range(len(coefficient)):
                EX2 = np.dot(np.dot(tensor_to_vec(O).T, khatri_rao([EZZT[0], EZZT[1],
                                                                    EZZT_t_pre[i]], reverse=bool)), np.ones([R * R, 1]))
                X_i = np.squeeze(tl.kruskal_tensor.kruskal_to_tensor([Z[0], Z[1], Z_pre[i]]))
                errs.append(np.dot(tensor_to_vec(L_pre[i]).T, tensor_to_vec(L_pre[i])) - 2 * tensor_to_vec(L_pre[i]).T \
                    .dot(tensor_to_vec(X_i)) + EX2)
                err += coefficient[i] * errs[i]
                nObs_sum += coefficient[i] * np.sum(O_pre[i])
            #  update noise tau
            a_tauN = (a_tau0 + 0.5 * nObs_sum)  # a_MnObs
            b_tauN = (b_tau0 + 0.5 * err)  # b_M

            tau = a_tauN / b_tauN

            # Update sparse matrix E
            sigma_E = np.reciprocal(tau + np.reciprocal(sigma_E0))
            E = (sigma_E * (E0 / sigma_E0 + tau * (C - np.expand_dims(X, axis=2))))

            temp1 = -0.5 * nObs * safelog(2 * pi) + 0.5 * nObs * (digamma(a_tauN) - safelog(b_tauN)) \
                    - 0.5 * tau * err
            for i in range(len(coefficient)):
                temp1 += coefficient[i] * (-0.5 * np.sum(O_pre[i]) * safelog(2 * pi) + 0.5 * nObs * (digamma(a_tauN) - safelog(b_tauN)) \
                    - 0.5 * tau * errs[i])
            temp2 = 0
            for n in range(N - 1):
                for i in range(dimY[n]):
                    temp2 += -0.5 * R * safelog(2 * pi) + 0.5 * safelog(det(ZSigma0[n][:, :, i])) - 0.5 * \
                             np.trace(ZSigma0[n][:, :, i].dot(ZSigma[n][:, :, i])) - 0.5 * \
                             np.dot(np.expand_dims(Z[n][i, :], 0), ZSigma0[n][:, :, i]).dot(
                                 np.expand_dims(Z[n][i, :], 1))

            temp2 += -0.5 * R * safelog(2 * pi) + 0.5 * safelog(det(np.squeeze(ZSigma0_t))) - 0.5 * \
                     np.trace(np.squeeze(ZSigma0_t).dot(np.squeeze(ZSigma_t))) - 0.5 * \
                     np.squeeze(np.dot(Z_t, np.squeeze(ZSigma0_t)).dot(np.reshape(Z_t, [R, 1])))

            temp3 = -safelog(gamma(a_tau0)) + a_tau0 * safelog(b_tau0) + (a_tau0 - 1) * (
                    digamma(a_tauN) - safelog(b_tauN)) \
                    - b_tau0 * (a_tauN / b_tauN)

            temp4 = 0
            for n in range(N - 1):
                for i in range(ZSigma[n].shape[2]):
                    temp4 += 0.5 * safelog(np.linalg.det(ZSigma[n][:, :, i])) + 0.5 * R * (1 + safelog(2 * pi))

            temp4 += 0.5 * safelog(np.linalg.det(ZSigma_t)) + 0.5 * R * (1 + safelog(2 * pi))

            temp5 = safelog(gamma(a_tauN)) - (a_tauN - 1) * digamma(a_tauN) - safelog(b_tauN) + a_tauN

            temp6 = -0.5 * nObs * safelog(2 * pi) - np.sum(
                0.5 * safelog(sigma_E0) - 0.5 * (np.square(E) + sigma_E - 2 * E * E + np.square(E0)) / sigma_E0)

            temp7 = 0.5 * nObs * safelog(2 * pi) + np.sum(
                0.5 * safelog(sigma_E) + 0.5 * (np.square(E) + sigma_E - 2 * E * E0 + np.square(E)) / sigma_E)
            LB.append(temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7)

            # Display progress
            if it > 2:
                LBRelChan = abs(LB[it] - 2 * LB[it - 1] + LB[it - 2]) / -LB[1]
            else:
                LBRelChan = 0

            if verbose:
                print('Iter. %d: RelChan = %g' % (it, LBRelChan))
            # Convergence check
            if it > 5 and (abs(LBRelChan) < tol):
                # print('======= Converged===========')
                break
    L = tl.kruskal_tensor.kruskal_to_tensor([Z[0], Z[1], Z_t])
    return Z, E, ZSigma, Z_t, ZSigma_t, L, EZZT, EZZT_t, a_tauN, b_tauN, tau


def BCPF_IC(Y, outliers_p, maxRank, maxiters, tol=1e-5, verbose=1, init='rand'):
    print(init)
    R = maxRank
    dimY = Y.shape
    N = Y.ndim
    T = dimY[2]
    outliers_count = np.sum(outliers_p, (0, 1))
    # Initialization
    a_tauN = 1e-6
    b_tauN = 1e-6

    tau = 1e4  # E[tau]
    dscale = 1

    Z = []
    ZSigma = []

    # Initialize model parameters
    for n in range(N-1):
        if init == 'rand':
            Z.append(np.random.randn(dimY[n], R))  # E[A ^ (n)]
        elif init == 'ml':
            U, S, _ = scipy.linalg.svd(unfold(Y, n), full_matrices=False)
            S = np.diag(S)
            Z.append(np.dot(U[:, :R], np.square(S[:R, :R])))
        ZSigma.append(np.zeros([R, R, dimY[n]]))  # covariance of A ^ (n)
        for i in range(dimY[n]):
            ZSigma[n][:, :, i] = np.eye(R)

    #U, S, _ = np.linalg.svd(unfold(Y, 2), full_matrices=False)
    #S = np.diag(S)
    #z_tmp = np.dot(U[:, :R], np.square(S[:R, :R]))

    # --------- E(aa') = cov(a,a) + E(a)E(a')----------------
    EZZT = []
    for n in range(N-1):
        EZZT.append(np.reshape(ZSigma[n], [R*R, dimY[n]], 'F').T)#向量化的B^(n)
    #         EZZT{n} = (reshape(ZSigma{n}, [R*R, dimY(n)]))' + khatrirao_fast(Z{n}',Z{n}')'\
    Z_ST = np.zeros([0, R])
    X = np.zeros([dimY[0], dimY[1], 0])

    RSE = []
    TPRS = []
    FPRS = []
    EZZT_ts = []
    count = 0
    false_locations = [] # Record OD pair that wrongly classified as outliers
    D = 288 # One day sampling span
    W = 2016 # One day sampling span

    for t in range(T):
        #print(t)
        #  Model learning
        coefficient = []
        EZZT_t_pre = []
        Z_pre = []
        O_pre = []
        L_pre = []
        for p, c in [(D+1, -1), (W, 1), (W+D, -1)]:#
            if t-p >= 0:
                coefficient.append(c)
                Z_pre.append(np.expand_dims(Z_ST[t-p, :], axis=0))
                O_pre.append(np.ones([dimY[0], dimY[1], 1]))
                EZZT_t_pre.append(EZZT_ts[t-p])
                L_pre.append(np.expand_dims(X[:, :, t-p], axis=2))
        params = {
            'Z': Z, 'ZSigma': ZSigma,
            'coefficient': coefficient,
             'EZZT': EZZT, 'EZZT_t_pre': EZZT_t_pre,
             'Z_pre': Z_pre, 'O_pre': O_pre,
             'a_tau0': a_tauN, 'b_tau0': b_tauN,
             'tau': tau, 'L_pre': L_pre}
        # update parameters
        Z, E, ZSigma, Z_t, ZSigma_t, L, EZZT, EZZT_t, a_tauN, b_tauN, tau = update(Y[:, :, t], np.ones([dimY[0], dimY[1], 1]), params, N, R, maxiters=maxiters)

        EZZT_ts.append(EZZT_t)
        Z_ST = np.concatenate([Z_ST, Z_t], axis=0)
        X = np.concatenate([X, L], axis=2)
        [TPR, FPR, locations] = check(E, outliers_p[:, :, t], outliers_count[t], dimY)
        false_locations.append(locations)
        count += TPR
        L = np.expand_dims(L, 2)
        RSE.append(norm(Y[:, :, t]-L-E) / norm(Y[:, :, t]))
        TPRS.append(TPR)
        FPRS.append(FPR)

    # Prepare the results
    Z.append(Z_ST)
    X = tl.kruskal_to_tensor(Z) * dscale

    model = {}
    # Output
    model['L'] = X
    model['RSE'] = RSE
    model['TPRS'] = TPRS
    model['FPRS'] = FPRS
    model['precision'] = count / T
    model['FPR'] = np.sum(FPRS) / T
    model['ZSigma'] = ZSigma
    model['false_locations'] = false_locations
    return model