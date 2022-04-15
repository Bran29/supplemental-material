import numpy
import pandas as pd
import numpy as np #Python Scientific Library
from scipy.stats import sem
from scipy import stats
import scipy
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import copy

from math import ceil, floor
from collections import defaultdict, Counter

import torch
from torch import rand, randn, cat, stack
# from botorch.test_functisouons.synthetic import Hartmann
import itertools
import random
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn import metrics
import torch.nn.functional as F
import math
def compute_volume(S1, S2, d=1, method="normal"):
    X = np.append(S1, S2,axis=1)

    Temp=np.zeros((S1.shape[0],d))
    S1 = S1.reshape(-1, d)
    S2 = S2.reshape(-1, d)
    X = X.reshape(-1, 2*d)

    if method == "log":
        v1 = np.linalg.slogdet(S1.T @ S1)[1]
        v2 = np.linalg.slogdet(S2.T @ S2)[1]
        v_all = np.linalg.slogdet(X.T @ X)[1]
    elif method == "rowleverage":
        k = max(X.shape[0], X.shape[1])
        XXt = X.T.dot(X)
        U, eig, Ut = scipy.linalg.svd(XXt)
        AnotkF2 = np.sum(eig[k:])
        ridge_kernel = U.dot(np.diag(1 / (eig + AnotkF2 / k))).dot(Ut)

        vd1 = []
        vd2 = []
        vd = []
        v1 = 0
        v2 = 0
        v_all = 0

        for i in range(S1.shape[0]):
            vd1.append(S1[i:i + 1, :] @ ridge_kernel @ S1[i:i + 1, :].T)
            vd2.append(S2[i:i + 1, :] @ ridge_kernel @ S2[i:i + 1, :].T)

            v1 += vd1[-1]
            v2 += vd2[-1]
        for i in range(2 * S1.shape[0]):
            vd.append(X[i:i + 1, :] @ ridge_kernel @ X[i:i + 1, :].T)
            v_all += vd[-1]
    elif method == "leverage":
        k = max(X.shape[0], X.shape[1])
        XXt = X.dot(X.T)
        U, eig, Ut = scipy.linalg.svd(XXt)
        AnotkF2 = np.sum(eig[k:])
        ridge_kernel = U.dot(np.diag(1 / (eig + AnotkF2 / k))).dot(Ut)

        vd = []
        v1 = 0
        v2 = 0
        v_all = 0
        for i in range(d):
            vd.append(X[:, i:i + 1].T @ ridge_kernel @ X[:, i:i + 1])
            v1 += vd[-1] * np.linalg.slogdet(S1[:, i:i + 1].T @ S1[:, i:i + 1])[1] / \
                  np.linalg.slogdet(X[:, i:i + 1].T @ X[:, i:i + 1])[1]
            v2 += vd[-1] * np.linalg.slogdet(S2[:, i:i + 1].T @ S2[:, i:i + 1])[1] / \
                  np.linalg.slogdet(X[:, i:i + 1].T @ X[:, i:i + 1])[1]
            # v1 += np.linalg.slogdet(S1[:,i:i+1].T @ S1[:,i:i+1])[1]/np.linalg.slogdet(X[:,i:i+1].T @ X[:,i:i+1])[1]
            # v2 += np.linalg.slogdet(S2[:,i:i+1].T @ S2[:,i:i+1])[1]/np.linalg.slogdet(X[:,i:i+1].T @ X[:,i:i+1])[1]
            v_all += vd[-1] * np.linalg.slogdet(X[:, i:i + 1].T @ X[:, i:i + 1])[1]
    elif method == "combine":
        k = max(X.shape[0], X.shape[1])
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

        vcol = []

        for i in range(X.shape[1]):
            vcol.append(X[:, i:i + 1].T @ ridge_kernel @ X[:, i:i + 1])

        # S1
        for i in range(S1.shape[0]):
            for j in range(S1.shape[1]):
                v1 += vrow[i] * vcol[j]

        # S2
        for i in range(S2.shape[0]):
            for j in range(S1.shape[1], S1.shape[1] + S2.shape[1]):
                v2 += vrow[i] * vcol[j]

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                v_all += vrow[i] * vcol[j]
    else:
        v1 = np.sqrt(np.linalg.det(S1.T @ S1) + 1e-8)
        v2 = np.sqrt(np.linalg.det(S2.T @ S2) + 1e-8)
        v_all = np.sqrt(np.linalg.det(X.T @ X) + 1e-8)
        # v1 = np.sqrt(np.linalg.det(S1 @ S1.T) + 1e-8)
        # v2 = np.sqrt(np.linalg.det(S2@ S2.T) + 1e-8)
        # v_all = np.sqrt(np.linalg.det(X @ X.T) + 1e-8)

        # v1 = np.linalg.slogdet(S1 @ S1.T)[1]
        # v2 = np.linalg.slogdet(S2@ S2.T)[1]
        # v_all = np.linalg.slogdet(X@ X.T)[1]

    return v1, v2, v_all
def compute_single_volume(S1, d=1, method="normal"):
    S1 = S1.reshape(-1, d)
    v1 = np.sqrt(np.linalg.det(S1.T @ S1) + 1e-8)
    return v1

def compute_X_tilde_and_counts_cluster(X, k_cluseter):
    D = X.shape[1]
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(X)

    kmeans = KMeans(
        init="random",
        n_clusters=int(k_cluseter),
        n_init=10,
        max_iter=300,
        random_state=42
        # random_state=42
    )


    kmeans.fit(X)

    X_tilde=kmeans.cluster_centers_
    # _, idx = np.unique(X_tilde, return_index=True)
    # X_tilde = X_tilde[np.sort(idx)]
    dist = cdist(X_tilde, X_tilde, metric='euclidean')
    # 测试
    # for i in range(1,X_tilde.shape[0]):
    #     X_tilde[i]=X_tilde[i]/X_tilde[0]

    cubes = Counter() # a dictionary to store the freqs

    for label in kmeans.labels_:
        cubes[label] += 1

    label=kmeans.labels_

    score=metrics.silhouette_score(X,label)
    # score=metrics.calinski_harabasz_score(X,label)
    return X_tilde, cubes,dist,score

def compute_X_tilde_and_counts(X, omega):
    """
    Compresses the original feature matrix X to  X_tilde with the specified omega.

    Returns:
       X_tilde: compressed np.ndarray
       cubes: a dictionary of cubes with the respective counts in each dcube
    """
    D = X.shape[1]
    assert 0 < omega <= 1, "omega must be within range [0,1]."
    m = ceil(1.0 / omega) # number of intervals for each dimension

    cubes = Counter() # a dictionary to store the freqs
    # key: (1,1,..)  a d-dimensional tuple, each entry between [0, m-1]
    # value: counts

    Omega = defaultdict(list)

    # a dictionary to store cubes of not full size
    for x in X:
        cube = []
        for d, xd in enumerate(x):
            d_index = floor(xd / omega)
            cube.append(d_index)

        cube_key = tuple(cube)
        cubes[cube_key] += 1

        Omega[cube_key].append(x)

        '''
        if cube_key in Omega:

            # Implementing mean() to compute the average of all rows which fall in the cube

            Omega[cube_key] = Omega[cube_key] * (1 - 1.0 / cubes[cube_key]) + 1.0 / cubes[cube_key] * x
            # Omega[cube_key].append(x)
        else:
             Omega[cube_key] = x
        '''
    X_tilde = stack([stack(list(value)).mean(axis=0) for key, value in Omega.items()])

    return X_tilde, cubes


def compute_volumes(datasets, d=1):
    d = datasets[0].shape[1]
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reshape(-1 ,d)

    X = np.concatenate(datasets, axis=0).reshape(-1, d)
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det(dataset.T @ dataset) + 1e-8)

    volume_all = np.sqrt(np.linalg.det(X.T @ X) + 1e-8)
    return volumes, volume_all

def compute_robust_volumes_ref(X_tildes, dcube_collections,s):
    # N = sum([len(X_tilde) for X_tilde in X_tildes])
    alpha = 1.0 / (10 * s)  # it means we set beta = 10
    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    volumes, volume_all = compute_volumes(X_tildes, d=X_tildes[0].shape[1])
    robust_volumes = np.zeros_like(volumes)
    for i, (volume, hypercubes) in enumerate(zip(volumes, dcube_collections)):
        rho_omega_prod = 1.0
        for cube_index, freq_count in hypercubes.items():
            # if freq_count == 1: continue # volume does not monotonically increase with omega
            # commenting this if will result in volume monotonically increasing with omega
            rho_omega = (1 - alpha ** (freq_count + 1)) / (1 - alpha)

            rho_omega_prod *= rho_omega

        robust_volumes[i] = (volume * rho_omega_prod)
    return robust_volumes

def compute_robust_volume_mine(X_tildes1, dcube_collections1,s):
    alpha = 1.0 / (10 * s)  # it means we set beta = 10
    beta = 1.0 / (10 * s)

    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    volumes1 = compute_single_volume(X_tildes1 ,d=X_tildes1.shape[1])

    rho_omega_prod = 1.0
    for cube_index, freq_count in dcube_collections1.items():
        # if freq_count == 1: continue # volume does not monotonically increase with omega
        # commenting this if will result in volume monotonically increasing with omega
        rho_omega = (1 - alpha ** (freq_count + 1)) / (1 - alpha)

        rho_omega_prod *= rho_omega

    robust_volumes1 = (volumes1 * rho_omega_prod)


    return robust_volumes1

def compute_robust_volumes_mine(X_tildes1, dcube_collections1,X_tildes2, dcube_collections2,s):
    alpha = 1.0 / (10 * s)  # it means we set beta = 10
    beta = 1.0 / (10 * s)

    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    volumes1, volumes2,volume_all = compute_volume(X_tildes1,X_tildes2 ,d=X_tildes1.shape[1])

    rho_omega_prod = 1.0
    for cube_index, freq_count in dcube_collections1.items():
        # if freq_count == 1: continue # volume does not monotonically increase with omega
        # commenting this if will result in volume monotonically increasing with omega
        rho_omega = (1 - alpha ** (freq_count + 1)) / (1 - alpha)

        rho_omega_prod *= rho_omega

    robust_volumes1 = (volumes1 * rho_omega_prod)


    for cube_index, freq_count in dcube_collections2.items():
        # if freq_count == 1: continue # volume does not monotonically increase with omega
        # commenting this if will result in volume monotonically increasing with omega
        rho_omega = (1 - alpha ** (freq_count + 1)) / (1 - alpha)

        rho_omega_prod *= rho_omega

    robust_volumes2  = (volumes2 * rho_omega_prod)
    return robust_volumes1,robust_volumes2

from collections import defaultdict

distortion_mean_mine = defaultdict(list)
distortion_std_mine = defaultdict(list)

distortion_mean_ref = defaultdict(list)
distortion_std_ref = defaultdict(list)

ds = [2]
M = 10
s_values = np.linspace(2000, 10000, 20)
deta=0.01
for d in ds:
    for s in s_values:
        cluster = min(int(math.pow(2,d)),2)
        distortions_mine = []
        distortions_ref = []
        s = int(s)
        for _ in tqdm(range(M)):

            norepli_S1 = randn((s, d)) * (0.2 - 0) + 0.8
            # repli_S1 = norepli_S1[random.randint(0,norepli_S1.shape[0]-1),:]
            # repli_S1 = repli_S1.repeat(int(0.1*(s)), 1)
            S1 = norepli_S1.repeat(10, 1)
            # S1=torch.vstack([norepli_S1,repli_S1])

            scaler = MinMaxScaler()
            S1 = scaler.fit_transform(S1)
            norepli_S1= scaler.fit_transform(norepli_S1)

            X_tilde1, cubes1,dist1,score1= compute_X_tilde_and_counts_cluster(S1, cluster)
            X_tilde_nrep, cubes_nrep, dist_nrep, score_nrep = compute_X_tilde_and_counts_cluster(norepli_S1, cluster)

            rv1 = compute_robust_volume_mine(X_tilde1, cubes1,2*s)
            rv1_nrefp=compute_robust_volume_mine(X_tilde_nrep, cubes_nrep,s)

            distortions_mine.append(rv1/rv1_nrefp)

            S1 = torch.from_numpy(S1)
            norepli_S1 = torch.from_numpy(norepli_S1)

            X_tilde, cubes = compute_X_tilde_and_counts(S1, omega=0.5)
            rv1 = compute_robust_volumes_ref([X_tilde], [cubes],S1.shape[0])[0]

            X_tilde, cubes = compute_X_tilde_and_counts(norepli_S1, omega=0.5)
            rv2 = compute_robust_volumes_ref([X_tilde], [cubes],norepli_S1.shape[0])[0]

            # if d == 1:
            #     X_tilde, cubes = compute_X_tilde_and_counts(S1, omega=1 / cluster)
            #     rv1 = compute_robust_volumes_ref([X_tilde], [cubes])[0]
            #
            #     X_tilde, cubes = compute_X_tilde_and_counts(S2, omega=1 / cluster)
            #     rv2 = compute_robust_volumes_ref([X_tilde], [cubes])[0]
            # else:
            #     X_tilde, cubes = compute_X_tilde_and_counts(S1, omega=1 / math.log(cluster, d))
            #     rv1 = compute_robust_volumes_ref([X_tilde], [cubes])[0]
            #
            #     X_tilde, cubes = compute_X_tilde_and_counts(S2, omega=1 / math.log(cluster, d))
            #     rv2 = compute_robust_volumes_ref([X_tilde], [cubes])[0]

            distortions_ref.append((rv1 / rv2))

        distortion_mean_mine[d].append(np.mean(distortions_mine))
        distortion_std_mine[d].append(np.std(distortions_mine))
        distortion_mean_ref[d].append(np.mean(distortions_ref))
        distortion_std_ref[d].append(np.std(distortions_ref))




plt.figure(figsize=(12, 7))
plt.axes(yscale = "log")

for i, d in enumerate(ds):
    plt.plot(s_values, distortion_mean_ref[d], linestyle = 'dashed', marker='^', label= "$\omega-based$ d="+str(d), linewidth=np.log(d+16))

    plt.plot(s_values, distortion_mean_mine[d], linestyle = 'solid', label= "$cluster-based$ d="+str(d), linewidth=np.log(d+16))

plt.legend(ncol=4,  fontsize=12)
plt.ylabel("Distortion $\\delta$", fontsize=30)
plt.xlabel("Size of $\mathbf{X}_S, \mathbf{X}_{S'}$", fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.title("Distortions vs. Size of $\mathbf{X}_S, \mathbf{X}_{S'}$", fontsize=32)
# plt.ylim(0.75, 1.15)
plt.hlines(xmin=s_values[0] ,xmax = s_values[-1],  y=np.exp(0), linewidth=3, linestyle='dotted', color='k')
# plt.hlines(xmin=s_values[0] ,xmax = s_values[-1],  y=1.0/ (np.exp(0.1)), linewidth=5, linestyle='dotted', color='k')

plt.tight_layout()
plt.show()

distortion_mean_mine=np.array(distortion_mean_mine)
distortion_std_mine=np.array(distortion_std_mine)
distortion_mean_ref=np.array(distortion_mean_ref)
distortion_std_ref=np.array(distortion_std_ref)

np.save("../datasave/infation_mean_mine2.npy", distortion_mean_mine)
np.save("../datasave/infation_std_mine2.npy", distortion_std_mine)
np.save("../datasave/infation_mean_ref2.npy", distortion_mean_ref)
np.save("../datasave/infation_std_ref2.npy", distortion_std_ref)

