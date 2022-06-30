import numpy
import pandas as pd
import numpy as np #Python Scientific Library
import scipy
from scipy.stats import sem
from scipy import stats

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model

import matplotlib.pyplot as plt
import copy

from math import ceil, floor
from collections import defaultdict, Counter

import torch
from torch import rand, randn, cat, stack
# from botorch.test_functisouons.synthetic import Hartmann
import itertools
import random

# def shannon_entropy(S):
#
# def total_correlation(S):
#
# def shapley(S,d):
#     svalue=np.zeros(1,d)
#     pre_svalue=svalue
#     t=0
#
#     permutations = list(itertools.permutations(range(d)))
#     random.shuffle(permutations)
#
#     while (numpy.sum(numpy.abs(svalue-pre_svalue)/numpy.abs(svalue))/d>=0.05):
#         permutation=permutations[t]
#         v=np.zeros(d)
#         for i in range(d):
#             if
#         t+=1
#     return
def column_shapley(S1,S2,d=1):
    # 计算各feature的shapley
    S1=[ S1[:,i] for i in range(d)]
    S2 = [S2[:,i] for i in range(d)]

    print()
def compute_volume(S1, S2, d=1, method="log"):
    X = np.append(S1, S2)

    S1 = S1.reshape(-1, d)
    S2 = S2.reshape(-1, d)
    X = X.reshape(-1, d)

    if method=="log":
        v1 = np.linalg.slogdet(S1.T @ S1)[1]
        v2 = np.linalg.slogdet(S2.T @ S2)[1]
        v_all = np.linalg.slogdet(X.T @ X)[1]
    elif method=="rowleverage":
        k = max(X.shape[0], X.shape[1])
        XXt = X.T.dot(X)
        U, eig, Ut = scipy.linalg.svd(XXt)
        AnotkF2 = np.sum(eig[k:])
        ridge_kernel = U.dot(np.diag(1 / (eig + AnotkF2 / k))).dot(Ut)

        vd1 = []
        vd2=[]
        vd=[]
        v1 = 0
        v2 = 0
        v_all = 0

        for i in range(S1.shape[0]):
            vd1.append(S1[i:i + 1,:] @ ridge_kernel @ S1[ i:i + 1,:].T)
            vd2.append(S2[i:i + 1, :] @ ridge_kernel @ S2[i:i + 1, :].T)

            v1 +=vd1[-1]
            v2 += vd2[-1]
        for i in range(2*S1.shape[0]):
            vd.append(X[i:i + 1, :] @ ridge_kernel @ X[i:i + 1, :].T)
            v_all += vd[-1]
    elif method=="leverage":
        k = max(X.shape[0], X.shape[1])
        XXt=X.dot(X.T)
        U, eig, Ut = scipy.linalg.svd(XXt)
        AnotkF2 = np.sum(eig[k:])
        ridge_kernel = U.dot(np.diag(1 / (eig + AnotkF2 / k))).dot(Ut)


        vd=[]
        v1=0
        v2=0
        v_all=0
        for i in range(d):
            vd.append(X[:,i:i+1].T @ ridge_kernel@X[:,i:i+1])
            v1 += vd[-1]*np.linalg.slogdet(S1[:,i:i+1].T @ S1[:,i:i+1])[1]/np.linalg.slogdet(X[:,i:i+1].T @ X[:,i:i+1])[1]
            v2 += vd[-1]*np.linalg.slogdet(S2[:,i:i+1].T @ S2[:,i:i+1])[1]/np.linalg.slogdet(X[:,i:i+1].T @ X[:,i:i+1])[1]
            # v1 += np.linalg.slogdet(S1[:,i:i+1].T @ S1[:,i:i+1])[1]/np.linalg.slogdet(X[:,i:i+1].T @ X[:,i:i+1])[1]
            # v2 += np.linalg.slogdet(S2[:,i:i+1].T @ S2[:,i:i+1])[1]/np.linalg.slogdet(X[:,i:i+1].T @ X[:,i:i+1])[1]
            v_all += vd[-1]*np.linalg.slogdet(X[:,i:i+1].T @ X[:,i:i+1])[1]
    elif method=="combine":
        k = max(X.shape[0],X.shape[1])
        XXt = X.T.dot(X)
        U, eig, Ut = scipy.linalg.svd(XXt)
        AnotkF2 = np.sum(eig[k:])
        ridge_kernel = U.dot(np.diag(1 / (eig + AnotkF2 / k))).dot(Ut)


        vrow = []
        v1 = 0
        v2 = 0
        v_all = 0

        for i in range(X.shape[0]):
            vrow.append(X[i:i + 1, :] @ ridge_kernel @ X[i:i + 1, :].T)

        XXt = X.dot(X.T)
        U, eig, Ut = scipy.linalg.svd(XXt)
        AnotkF2 = np.sum(eig[k:])
        ridge_kernel = U.dot(np.diag(1 / (eig + AnotkF2 / k))).dot(Ut)

        vcol=[]

        for i in range(X.shape[1]):
            vcol.append(X[:, i:i + 1].T @ ridge_kernel @ X[:, i:i + 1])

        # S1
        for i in range(S1.shape[0]):
            for j in range(S1.shape[1]):
                v1+=vrow[i]*vcol[j]

        #S2
        for i in range(S1.shape[0],S1.shape[0]+S2.shape[0]):
            for j in range(S1.shape[1]):
                v2+=vrow[i]*vcol[j]

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                v_all+=vrow[i]*vcol[j]
    else:
        v1 = np.sqrt(np.linalg.det(S1.T @ S1) + 1e-8)
        v2 = np.sqrt(np.linalg.det(S2.T @ S2) + 1e-8)
        v_all = np.sqrt(np.linalg.det(X.T @ X) + 1e-8)
        # v1 = np.sqrt(np.linalg.det(S1 @ S1.T) + 1e-8)
        # v2 = np.sqrt(np.linalg.det(S2 @ S2.T) + 1e-8)
        # v_all = np.sqrt(np.linalg.det(X @ X.T) + 1e-8)
    return v1, v2, v_all



def compute_bias(S1, S2, d=1,s=1):
    X = np.append(S1, S2)

    # S1 = S1.reshape(-1, d)
    # S2 = S2.reshape(-1, d)
    X = X.reshape(-1, d)
    #
    # # print("data matrices shapes:", S1.shape, S2.shape, X.shape)
    # XI_S1 = np.zeros( (X.shape[0], X.shape[0]) )
    # XI_S2 = np.zeros( (X.shape[0], X.shape[0]) )
    #
    # IS1 = np.append(np.ones(s), np.zeros(s))
    # IS2 = np.append(np.zeros(s), np.ones(s))
    # for i in range(X.shape[0]):
    #     XI_S1[i,i] = IS1[i]
    #     XI_S2[i,i] = IS2[i]
    #
    # XI_S1 = XI_S1 @ X
    # XI_S2 = XI_S2 @ X

    # S1_pinv, S2_pinv = np.linalg.pinv(XI_S1), np.linalg.pinv(XI_S2)
    S1_pinv, S2_pinv = np.linalg.pinv(S1), np.linalg.pinv(S2)
    X_pinv = np.linalg.pinv(X)
    return  np.linalg.norm(S1_pinv - X_pinv[:,0:S1.shape[0]]),np.linalg.norm(S2_pinv - X_pinv[:,S1.shape[0]:2*S1.shape[0]])


def compute_loss(S1, S2, f, d, param=False, Lambda=0):
    assert Lambda >= 0

    y1 = np.asarray([f(s1) for s1 in S1 ])
    y2 = np.asarray([f(s2) for s2 in S2 ])
    # y1 = f(S1)
    # y2 = f(S2)

    X = np.append(S1, S2)
    y = np.append(y1, y2).reshape(-1, 1)

    S1 = S1.reshape(-1, d)
    S2 = S2.reshape(-1, d)
    X = X.reshape(-1, d)

    if Lambda != 0:
        XI_S1_pinv = np.linalg.inv(S1.T @ S1 + Lambda * np.eye(d)) @ S1.T
        XI_S2_pinv = np.linalg.inv(S2.T @ S2 + Lambda * np.eye(d)) @ S2.T
        X_pinv = np.linalg.inv(X.T @ X + Lambda * np.eye(d)) @ X.T

    else:
        XI_S1_pinv, XI_S2_pinv = np.linalg.pinv(S1), np.linalg.pinv(S2)
        X_pinv = np.linalg.pinv(X)

    w_S1 = XI_S1_pinv @ y1
    w_S2 = XI_S2_pinv @ y2
    w_X =  X_pinv @ y

    loss1 = np.linalg.norm( X @ w_S1 - y )
    loss2 = np.linalg.norm( X @ w_S2 - y )
    loss = np.linalg.norm( X @ w_X - y )
    if not param:
        return loss1/len(y), loss2/len(y), loss/len(y)
    else:
        return loss1/len(y), loss2/len(y), loss/len(y), w_S1, w_S2


def compute_loss_dual(S1, S2, f, d, weight=False, Lambda=0):
    assert Lambda >= 0
    y1 = np.asarray([f(s1) for s1 in S1 ])
    y2 = np.asarray([f(s2) for s2 in S2 ])
    # y1 = f(S1)
    # y2 = f(S2)

    X = np.append(S1, S2)
    y = np.append(y1, y2).reshape(-1, 1)

    S1 = S1.reshape(-1, d)
    S2 = S2.reshape(-1, d)
    X = X.reshape(-1, d)

    if Lambda != 0:
        a1 = np.linalg.inv(S1 @ S1.T + Lambda *np.eye(len(S1))) @ y1
        a2 = np.linalg.inv(S2 @ S2.T + Lambda *np.eye(len(S2))) @ y2
        a_all = np.linalg.inv(X@ X.T + Lambda *np.eye(len(X))) @ y
    else:
        a1, a2 = np.linalg.inv(S1 @ S1.T) @ y1, np.linalg.inv(S2 @ S2.T) @ y2
        a_all = np.linalg.inv(X @ X.T) @ y


    loss1 = np.linalg.norm( (X @ S1.T)@ a1 - y)
    loss2 = np.linalg.norm( (X @ S2.T)@ a2 - y)
    loss = np.linalg.norm( (X @ X.T) @ a_all - y)

    if not weight:
        return loss1/len(y), loss2/len(y), loss/len(y)
    else:
        return loss1/len(y), loss2/len(y), loss/len(y), a1, a2


# Volume vs Bias
def VB():
    ds = [ 10, 20]
    M = 100
    s_values = np.linspace(100, 10000, 20)

    counts_bias_nleverage = []
    counts_bias_leverage = []

    plt.figure()

    for i,d in enumerate(ds):
        counts_nleverage = []
        counts_leverage = []
        for s in s_values:
            count_nleverage = 0.0
            count_leverage = 0.0
            s = int(s)
            for m in range(M):
                S1 = np.random.uniform(0, 1, (s, d))
                S2 = np.random.uniform(0, 1, (s, d))

                v1, v2, v_all = compute_volume(S1, S2, d,method="normal")
                (b1, b2) = compute_bias(S1, S2, d,s)

                if (v1 >= v2 and b1 <= b2) or (v1 <= v2 and b1 >= b2):
                    count_nleverage += 1



                # v1, v2, v_all = compute_volume(S1, S2, d, method="combine")
                # (b1, b2) = compute_bias(S1, S2, d, s)
                #
                # if (v1 >= v2 and b1 <= b2) or (v1 <= v2 and b1 >= b2):
                #     count_leverage += 1

                print(m, " ", s, " ", d)
            counts_nleverage.append(count_nleverage / M)
            counts_leverage.append(count_leverage / M)
        counts_bias_nleverage.append(counts_nleverage)
        counts_bias_leverage.append(counts_leverage)

        counts = counts_bias_nleverage[i]
        plt.plot(s_values, counts, linestyle='solid',  label="Volume d=" + str(d),
                 linewidth=np.log(d + 16))

        # counts = counts_bias_leverage[i]
        # plt.plot(s_values, counts, linestyle='solid', label="Ours d=" + str(d), linewidth=np.log(d + 16))
    counts_bias_nleverage=np.array(counts_bias_nleverage)
    np.save("../datasave/row_vb_uu.npy", counts_bias_nleverage)

    plt.legend()
    plt.ylabel("Percentage of times", fontsize=30)
    plt.xlabel("Size of $\mathbf{X}_S, \mathbf{X}_{S'}$", fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Volume vs. Bias', fontsize=32)
    plt.tight_layout()
    plt.show()

# Volume vs MSE
def VM():
    ds =  [5]
    M = 100
    s_values = np.linspace(10, 500, 20)

    counts_loss_nleverage = []
    counts_loss_leverage = []

    plt.figure()

    for i,d in enumerate(ds):
        coefs_in = np.random.uniform(0, 2, (1, d))

        fcn = lambda x: np.sin(np.dot(coefs_in, x))

        counts_nleverage = []
        counts_leverage = []

        for s in s_values:
            count_nleverage = 0.0
            count_leverage = 0.0
            s = int(s)

            for m in range(M):
                S1 = np.random.normal(0, 1, (s, d))
                S2 = np.random.normal(0, 1, (s, d))
                v1, v2, v_all = compute_volume(S1, S2, d,combine=True)
                l1, l2, loss = compute_loss(S1, S2, fcn, d)
                if (v1 >= v2 and l1 <= l2) or (v1 <= v2 and l1 >= l2):
                    count_nleverage += 1

                v1, v2, v_all = compute_volume(S1, S2, d,leverage=True)
                l1, l2, loss = compute_loss(S1, S2, fcn, d)
                if (v1 >= v2 and l1 <= l2) or (v1 <= v2 and l1 >= l2):
                    count_leverage += 1
                print(m, " ", s, " ", d)
            counts_nleverage.append(count_nleverage / M)
            counts_leverage.append(count_leverage/M)
        counts_loss_nleverage.append(counts_nleverage)
        counts_loss_leverage.append(counts_leverage)


        counts = counts_loss_nleverage[i]
        plt.plot(s_values, counts, linestyle='dashed', marker='^', label="Volume d=" + str(d),
                 linewidth=np.log(d + 16))

        counts = counts_loss_leverage[i]
        plt.plot(s_values, counts, linestyle='solid', label="Ours d=" + str(d), linewidth=np.log(d + 16))

    plt.legend()
    plt.ylabel("Percentage of times", fontsize=30)
    plt.xlabel("Size of $\mathbf{X}_S, \mathbf{X}_{S'}$", fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Volume vs. MSE', fontsize=32)

    # plt.hlines(xmin=0 ,xmax = 500,  y=0.5, linewidth=5, linestyle='dotted', color='k')
    plt.tight_layout()
    plt.show()

# Volume vs MSE
def VW():
    ds = [1,2, 5, 10]
    M = 50
    s_values = np.linspace(10, 500, 20)

    plt.figure(figsize=(12, 5.5))

    gauss_counts_loss = []
    uniform_counts_loss = []
    for i,d in ds:

        coefs_in = np.random.uniform(0, 2, (1, 2*d))
        coefs_out = np.random.uniform(-1, 1, (1, 2*d))

        fcn = lambda x: np.sin(np.dot(coefs_in, x))

        counts = []
        v1s, v2s, vs = [], [], []
        for s in s_values:
            count = 0.0
            s = int(s)
            V1, V2, V = 0,0,0
            for _ in range(M):
                S1 = np.random.normal(0, 1, (s, d))
                S2 = np.random.normal(0, 1, (s, d))

                v1, v2, v_all = compute_volume(S1, S2, d)
                l1, l2 = compute_w(S1, S2, fcn, d)
                if (v1 >= v2 and l1 <= l2) or (v1 <= v2 and l1 >= l2):
                    count += 1
            counts.append(count/M)
        gauss_counts_loss.append(counts)

        counts = []
        v1s, v2s, vs = [], [], []
        for s in s_values:
            count = 0.0
            s = int(s)

            V1, V2, V = 0,0,0
            for _ in range(M):
                S1 = np.random.uniform(0, 1, (s, d))
                S2 = np.random.uniform(0, 1, (s, d))

                v1, v2, v_all = compute_volume(S1, S2, d)
                l1, l2 = compute_w(S1, S2, fcn, d)
                if (v1 >= v2 and l1 <= l2) or (v1 <= v2 and l1 >= l2):
                    count += 1

            counts.append(count/M)
        uniform_counts_loss.append(counts)

        counts = gauss_counts_loss[i]
        plt.plot(s_values, counts, linestyle='dashed', marker='^', label="Volume d=" + str(d),
                 linewidth=np.log(d + 16))

        counts = uniform_counts_loss[i]
        plt.plot(s_values, counts, linestyle='solid', label="Ours d=" + str(d), linewidth=np.log(d + 16))


    plt.legend(ncol=4, loc='lower center', fontsize=22)
    plt.ylabel("Percentage of times", fontsize=30)
    plt.xlabel("Size of $\mathbf{X}_S, \mathbf{X}_{S'}$", fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Volume vs. MSE', fontsize=32)
    plt.ylim(0.25)
    # plt.hlines(xmin=0 ,xmax = 500,  y=0.5, linewidth=5, linestyle='dotted', color='k')
    plt.tight_layout()
    plt.show()

def VBM():
    ds =  [1,2,5,10,20]
    M = 100
    s_values = np.linspace(10, 500, 20)

    counts_method1_b = []
    counts_method2_b = []
    counts_method1_m = []
    counts_method2_m= []


    plt.figure(figsize=(20,8))
    ax1 = plt.gca()

    plt.figure(figsize=(20,8))
    ax2 = plt.gca()

    for i,d in enumerate(ds):
        coefs_in = np.random.uniform(0, 2, (1, d))

        fcn = lambda x: np.sin(np.dot(coefs_in, x))

        counts_method1_d_b = []
        counts_method2_d_b = []
        counts_method1_d_m = []
        counts_method2_d_m = []

        for s in s_values:
            count_method1_ds_b = 0
            count_method2_ds_b =0
            count_method1_ds_m = 0
            count_method2_ds_m = 0
            s = int(s)

            for m in range(M):
                S1 = np.random.normal(0, 1, (s, d))
                S2 = np.random.normal(0, 1, (s, d))

                v1, v2, v_all = compute_volume(S1, S2, d, method="combine")

                (b1, b2) = compute_bias(S1, S2, d, s)
                if (v1 >= v2 and b1 <= b2) or (v1 <= v2 and b1 >= b2):
                    count_method1_ds_b += 1

                l1, l2, loss = compute_loss(S1, S2, fcn, d)
                if (v1 >= v2 and l1 <= l2) or (v1 <= v2 and l1 >= l2):
                    count_method1_ds_m += 1

                v1, v2, v_all = compute_volume(S1, S2, d,method="leverage")

                (b1, b2) = compute_bias(S1, S2, d, s)
                if (v1 >= v2 and b1 <= b2) or (v1 <= v2 and b1 >= b2):
                    count_method2_ds_b += 1

                l1, l2, loss = compute_loss(S1, S2, fcn, d)
                if (v1 >= v2 and l1 <= l2) or (v1 <= v2 and l1 >= l2):
                    count_method2_ds_m += 1
                print(m, " ", s, " ", d)
            counts_method1_d_b.append(count_method1_ds_b / M)
            counts_method1_d_m.append(count_method1_ds_m/M)
            counts_method2_d_b.append(count_method2_ds_b / M)
            counts_method2_d_m.append(count_method2_ds_m / M)
        counts_method1_b.append(counts_method1_d_b)
        counts_method1_m.append(counts_method1_d_m)
        counts_method2_b.append(counts_method2_d_b)
        counts_method2_m.append(counts_method2_d_m)

        plt.sca(ax1)
        counts = counts_method1_b[i]
        plt.plot(s_values, counts, linestyle='dashed', marker='^', label="Volume d=" + str(d),
                 linewidth=np.log(d + 16))

        counts = counts_method2_b[i]
        plt.plot(s_values, counts, linestyle='solid', label="Ours d=" + str(d), linewidth=np.log(d + 16))

        plt.sca(ax2)
        counts = counts_method1_m[i]
        plt.plot(s_values, counts, linestyle='dashed', marker='^', label="Volume d=" + str(d),
                 linewidth=np.log(d + 16))

        counts = counts_method2_m[i]
        plt.plot(s_values, counts, linestyle='solid', label="Ours d=" + str(d), linewidth=np.log(d + 16))

    plt.sca(ax1)
    plt.legend()
    plt.ylabel("Percentage of times", fontsize=30)
    plt.xlabel("Size of $\mathbf{X}_S, \mathbf{X}_{S'}$", fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Volume vs. Bias', fontsize=32)

    # plt.hlines(xmin=0 ,xmax = 500,  y=0.5, linewidth=5, linestyle='dotted', color='k')
    plt.tight_layout()

    plt.sca(ax2)
    plt.legend()
    plt.ylabel("Percentage of times", fontsize=30)
    plt.xlabel("Size of $\mathbf{X}_S, \mathbf{X}_{S'}$", fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Volume vs. MSE', fontsize=32)

    # plt.hlines(xmin=0 ,xmax = 500,  y=0.5, linewidth=5, linestyle='dotted', color='k')
    plt.tight_layout()
    plt.show()

VB()