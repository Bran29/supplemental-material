import random

import numpy as np
import torch
import copy
import pandas as pd
from itertools import permutations
from math import factorial

from main_utils import get_synthetic_datasets, generate_linear_labels, friedman_function, hartmann_function, scale_normal
from data_utils import load_used_car, load_uber_lyft, load_credit_card, load_hotel_reviews
from volume import replicate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale
import NN_Classification_Wine_Quality as classification
import NN_Regression_Wine_Quality as regression
# Reproducebility
seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ---------- DATA PREPARATION ----------
D=10
M=3
df=pd.read_csv("../data/wine_quality/WineQT.csv")
df.drop('Id',axis=1,inplace=True)
df["quality"]-=df["quality"].min()

X=df.drop(columns=["quality","pH"])
y=df[["pH"]]

# y = minmax_scale(y)
# X = X.to_numpy()
# y = y.to_numpy()
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.1)

feature_datasets=[]
label_datasets=[]
Class_Num=6

X_train200=X_train[0:200].to_numpy()
X_train400=np.repeat(X_train200,repeats=2,axis=0)
X_train800=np.repeat(X_train200,repeats=4,axis=0)

y_train200=y_train[0:200].to_numpy()
y_train400=np.repeat(y_train200,repeats=2,axis=0)
y_train800=np.repeat(y_train200,repeats=4,axis=0)

feature_datasets=[X_train200,X_train400,X_train800]
label_datasets=[y_train200,y_train400,y_train800]


# ---------- DATA VALUATIONS ----------
res = {}

"""
Direct Volume-based values

"""
from volume import *

# train_features = [dataset.data for dataset in train_datasets]
volumes, vol_all = compute_volumes(feature_datasets, D)

volumes_all = np.asarray(list(volumes) + [vol_all])
print('-------Volume Statistics ------')


print("Original volumes: ", volumes, "volume all:", vol_all)
res['vol'] = volumes

"""
Discretized Robust Volume-based Shapley values
"""
import random
def shapley_volume(Xs, omega=0.1):
    M = len(Xs)
    orderings = list(permutations(range(M)))

    s_values = torch.zeros(M)
    monte_carlo_s_values = torch.zeros(M)

    s_value_robust = torch.zeros(M)
    monts_carlo_s_values_robust = torch.zeros(M)

    s_value_cluster_robust = torch.zeros(M)
    monts_carlo_s_values_cluster_robust = torch.zeros(M)

    # Monte-carlo : shuffling the ordering and taking the first K orderings
    random.shuffle(orderings)
    K = 4 # number of permutations to sample
    for ordering_count, ordering in enumerate(orderings):

        prefix_vol = 0
        prefix_robust_vol = 0
        prefix_cluster_robust_vol=0
        for position, i in enumerate(ordering):
            # volume sv
            curr_indices = set(ordering[:position+1])

            curr_train_X = torch.cat([torch.from_numpy(dataset) for j, dataset in enumerate(Xs) if j in curr_indices]).reshape(-1, D)

            curr_vol = torch.sqrt(torch.linalg.det(curr_train_X.T @ curr_train_X) + 1e-8)


            marginal = curr_vol - prefix_vol
            prefix_vol = curr_vol
            s_values[i] += marginal

            # w robust sv
            X_tilde, cubes = compute_X_tilde_and_counts(curr_train_X, omega)

            robust_vol = compute_robust_volumes([X_tilde], [cubes])[0]


            marginal_robust = robust_vol - prefix_robust_vol
            s_value_robust[i] += marginal_robust
            prefix_robust_vol = robust_vol
            #cluster-based
            robust_volumes = []
            Xtildes, cluster, dist, score = compute_X_tilde_and_counts_cluster(curr_train_X, k_cluseter=6)
            cluster_robust_vol=compute_robust_volumes_mine(Xtildes, cluster, Xtildes.shape[0])
            marginal_cluster_robust = cluster_robust_vol - prefix_cluster_robust_vol
            s_value_cluster_robust[i] += marginal_cluster_robust
            prefix_cluster_robust_vol = cluster_robust_vol

            if ordering_count < K:
                monte_carlo_s_values[i] += marginal
                monts_carlo_s_values_robust[i] += marginal_robust
                monts_carlo_s_values_cluster_robust[i] += marginal_cluster_robust
    s_values /= factorial(M)
    s_value_robust /= factorial(M)
    s_value_cluster_robust /= factorial(M)
    monte_carlo_s_values /= K
    monts_carlo_s_values_robust /= K
    monts_carlo_s_values_cluster_robust /= K

    print('------Volume-based Shapley value Statistics ------')
    print("Volume-based Shapley values:", s_values)
    print("Robust Volume Shapley values:", s_value_robust)
    print("Cluster Robust Volume Shapley values:", s_value_cluster_robust)
    print("Volume-based MC-Shapley values:", monte_carlo_s_values)
    print("Robust Volume MC-Shapley values:", monts_carlo_s_values_robust)
    print("Cluster Robust Volume MC-Shapley values:", monts_carlo_s_values_cluster_robust)
    print('-------------------------------------')
    return s_values, s_value_robust,s_value_cluster_robust, monte_carlo_s_values, monts_carlo_s_values_robust,monts_carlo_s_values_cluster_robust


# feature_datasets_include_all = copy.deepcopy(feature_datasets) + [torch.vstack(feature_datasets) ]

# s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust = shapley_volume(feature_datasets, omega=0.5, alpha=alpha)
s_values, s_value_robust,s_value_cluster_robust, monte_carlo_s_values, monts_carlo_s_values_robust ,monts_carlo_s_values_cluster_robust= shapley_volume(feature_datasets, omega=0.1)

# omega_res_rv = []
# omega_res_rvsv = []
# omega_res_vsv = []
# omega_upper = 0.5
# for omega in np.linspace(0.001,omega_upper,30):
#     Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets))
#     Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
#     robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)
#     rv = np.array(robust_volumes)
#     omega_res_rv.append(rv/np.sum(rv))
#     s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust = shapley_volume(feature_datasets, omega=omega)
#     rvsv = np.array(s_value_robust)
#     rvsv[rvsv < 0] = 0
#     omega_res_rvsv.append(rvsv/np.sum(rvsv))
#     vsv = np.array(s_values)
#     omega_res_vsv.append(vsv/np.sum(vsv))
# omega_res_rv = np.array(omega_res_rv)
# omega_res_rvsv = np.array(omega_res_rvsv)
# omega_res_vsv = np.array(omega_res_vsv)
# np.savez('outputs/omega_exp_disjoint_normal_{}.npz'.format(omega_upper), omega_res_rv=omega_res_rv, omega_res_rvsv=omega_res_rvsv, omega_res_vsv=omega_res_vsv)


res['vol_sv'], res['vol_sv_robust'], res['vol_sv_cluster_robust']  = s_values, s_value_robust,s_value_cluster_robust
res['vol_mc_sv'], res['vol_mc_sv_robust'],res['vol_mc_sv_cluster_robust'] = monte_carlo_s_values, monts_carlo_s_values_robust,monts_carlo_s_values_cluster_robust


"""
regression

"""

#  in regression
valid_loss_all = 100


train_X = np.vstack(feature_datasets).reshape(-1, D)
train_y = np.vstack(label_datasets)

test_X = X_test.to_numpy()
test_y = y_test.to_numpy()

for r in range(20):
    valid_loss_all_r, valid_loss_all_old = regression.loss_result(train_X, train_y, test_X, test_y)
    valid_loss_all = min(valid_loss_all_r, valid_loss_all)

loo_values = np.zeros(M)
loo_losses =[]
for i in range(M):
    loo_dataset = []
    loo_label = []

    for j, (dataset, label) in enumerate(zip(feature_datasets, label_datasets)):
        if i == j: continue

        loo_dataset.append(dataset)
        loo_label.append(label)

    loo_dataset = np.vstack(loo_dataset).reshape(-1, D)
    loo_label = np.vstack(loo_label)
    loo_test_loss = 0
    for r in range(20):
        loo_test_loss_r, _ = regression.loss_result(loo_dataset, loo_label, test_X, test_y)
        loo_test_loss += loo_test_loss_r
    loo_test_loss /= 20
    loo_losses.append(loo_test_loss)
    loo_values[i]=loo_test_loss - valid_loss_all

print('------Leave One Out Statistics ------')
print("Full test loss:", valid_loss_all)
print("Leave-one-out test losses:", loo_losses)
print("Leave-one-out (loo_loss - full_loss) :", loo_values)
print('-------------------------------------')

res['loo_reg'] = loo_values

from itertools import permutations
from math import factorial

orderings = list(permutations(range(M)))

s_values = np.zeros(M)
monte_carlo_s_values = np.zeros(M)

# Monte-carlo : shuffling the ordering and taking the first K orderings
np.random.shuffle(orderings)
K = 4  # number of permutations to sample

test_loss = valid_loss_all

truncated_mc_s_values = np.zeros(M)

tol = 0.001

# Random initilaization, needed to compute marginal against empty set

init_test_loss = valid_loss_all_old

for ordering_count, ordering in enumerate(orderings):

    prefix_test_losses = []
    for position, i in enumerate(ordering):

        curr_indices = set(ordering[:position + 1])

        curr_train_X = np.vstack([dataset for j, dataset in enumerate(feature_datasets) if j in curr_indices]).reshape(
            -1, D)
        curr_train_y = np.vstack([label for j, label in enumerate(label_datasets) if j in curr_indices])

        # curr_pinv_ = np.linalg.pinv(curr_train_X)
        curr_test_loss = 0
        for r in range(5):
            curr_test_loss_r, _ = regression.loss_result(curr_train_X, curr_train_y, test_X, test_y)
            curr_test_loss += curr_test_loss_r
        curr_test_loss /= 5

        if position == 0:  # first in the ordering
            marginal = init_test_loss - curr_test_loss
        else:
            marginal = prefix_test_losses[-1] - curr_test_loss
        s_values[i] += marginal

        prefix_test_losses.append(curr_test_loss)

        if ordering_count < K:
            monte_carlo_s_values[i] += marginal

            if np.abs(test_loss - curr_test_loss) > tol or ordering_count == 0:
                truncated_mc_s_values[i] += marginal

s_values /= factorial(M)
monte_carlo_s_values /= K
truncated_mc_s_values /= K
print('------Test loss Shapley value Statistics ------')
print("Test loss-based Shapley values:", s_values)
print("Test loss-based MC-Shapley values:", monte_carlo_s_values)
print("Test loss-based TMC-Shapley values:", truncated_mc_s_values)
print('-------------------------------------')

res['loss_sv_reg'], res['loss_mc_sv_reg'], res['loss_tmc_sv_reg'] = s_values, monte_carlo_s_values, truncated_mc_s_values
from scipy.stats import sem
import random

have_gp = True
if have_gp:
    from gpytorch_ig import compute_IG, fit_model

    trials = 5

    s_values_IG_trials = []
    mc_s_values_IG_trials = []

    for t in range(trials):
        all_train_X = np.vstack(feature_datasets).astype(np.float32)
        all_train_y = np.vstack(label_datasets).astype(np.float32).reshape(-1 ,1).squeeze()
        joint_model, joint_likelihood = fit_model(torch.from_numpy(all_train_X), torch.from_numpy(all_train_y))

        s_values_IG = torch.zeros(M)
        monte_carlo_s_values_IG = torch.zeros(M)

        orderings = list(permutations(range(M)))
        # Monte-carlo : shuffling the ordering and taking the first K orderings
        random.shuffle(orderings)
        K = 4  # number of permutations to sample

        for ordering_count, ordering in enumerate(orderings):

            prefix_IGs = []
            for position, i in enumerate(ordering):

                curr_indices = set(ordering[:position + 1])

                curr_train_X = np.vstack(
                    [dataset for j, dataset in enumerate(feature_datasets) if j in curr_indices]).reshape(-1, D)
                # curr_train_y = torch.cat([label for j, label in enumerate(labels)  if j in curr_indices ]).reshape(-1, 1)
                # curr_train_y = curr_train_y.squeeze()
                # curr_train_X, curr_train_y = torch.from_numpy(curr_train_X), torch.from_numpy(curr_train_y).squeeze()

                # NO NEED TO RETRAIN
                # model, likelihood = fit_model(curr_train_X, curr_train_y)
                # curr_IG = compute_IG(all_train_X, model, likelihood)

                curr_IG = compute_IG(torch.from_numpy(curr_train_X), joint_model, joint_likelihood)

                if position == 0:  # first in the ordering
                    marginal = curr_IG - 0
                else:
                    marginal = curr_IG - prefix_IGs[-1]
                s_values_IG[i] += marginal
                prefix_IGs.append(curr_IG)

                if ordering_count < K:
                    monte_carlo_s_values_IG[i] += marginal

        s_values_IG /= factorial(M)
        monte_carlo_s_values_IG /= K

        s_values_IG_trials.append(s_values_IG)
        mc_s_values_IG_trials.append(monte_carlo_s_values_IG)

    s_values_IG_trials = torch.stack(s_values_IG_trials)
    mc_s_values_IG_trials = torch.stack(mc_s_values_IG_trials)

    print('------Information Gain Shapley value Statistics ------')
    print("IG-based Shapley values: mean {}, sem {}".format(torch.mean(s_values_IG_trials, 0),
                                                            sem(s_values_IG_trials, axis=0)))
    print("IG-based MC-Shapley values: mean {}, sem {}".format(torch.mean(mc_s_values_IG_trials, 0),
                                                               sem(mc_s_values_IG_trials, axis=0)))
    print('-------------------------------------')

    res['ig_sv_reg'], res['ig_mc_sv_reg'] = torch.mean(s_values_IG_trials, 0), torch.mean(mc_s_values_IG_trials, 0)

# have_spgp = False
# if have_spgp:
#     from gpytorch_ig import compute_IG, fit_model
#
#     trials = 5
#
#     s_values_IG_trials = []
#     mc_s_values_IG_trials = []
#
#     for t in range(trials):
#
#         inducing_ratio = 0.25
#         inducing_count = int(torch.sum(torch.tensor(train_sizes)) * inducing_ratio * M)
#
#         end, begin = 1, 0
#         # uniform distribution of inducing
#         inducing_points = torch.rand((inducing_count, D)) * (end - begin) + begin
#
#         all_train_X = torch.cat(feature_datasets)
#         all_train_y = torch.cat(labels).reshape(-1 ,1).squeeze()
#         joint_model, joint_likelihood = fit_model(all_train_X, all_train_y, inducing_points=inducing_points)
#
#
#         s_values_IG = torch.zeros(M)
#         monte_carlo_s_values_IG = torch.zeros(M)
#
#         orderings = list(permutations(range(M)))
#         # Monte-carlo : shuffling the ordering and taking the first K orderings
#         random.shuffle(orderings)
#         K = 4 # number of permutations to sample
#
#         for ordering_count, ordering in enumerate(orderings):
#
#             prefix_IGs = []
#             for position, i in enumerate(ordering):
#
#                 curr_indices = set(ordering[:position+1])
#
#                 curr_train_X = torch.cat([dataset for j, dataset in enumerate(feature_datasets)  if j in curr_indices ]).reshape(-1, D)
#                 # curr_train_y = torch.cat([label for j, label in enumerate(labels)  if j in curr_indices ]).reshape(-1, 1)
#                 # curr_train_y = curr_train_y.squeeze()
#
#                 # model, likelihood = fit_model(curr_train_X, curr_train_y, inducing_points=inducing_points)
#                 curr_IG = compute_IG(curr_train_X, joint_model, joint_likelihood)
#
#                 if position == 0: # first in the ordering
#                     marginal = curr_IG  - 0
#                 else:
#                     marginal = curr_IG - prefix_IGs[-1]
#                 s_values_IG[i] += marginal
#                 prefix_IGs.append(curr_IG)
#
#                 if ordering_count < K:
#                     monte_carlo_s_values_IG[i] += marginal
#
#         s_values_IG /= factorial(M)
#         monte_carlo_s_values_IG /= K
#
#         s_values_IG_trials.append(s_values_IG)
#         mc_s_values_IG_trials.append(monte_carlo_s_values_IG)
#
#
#     s_values_IG_trials = torch.stack(s_values_IG_trials)
#     mc_s_values_IG_trials = torch.stack(mc_s_values_IG_trials)
#
#     print('------Information Gain SPGP Shapley value Statistics ------')
#     print("SPGP IG-based Shapley values: mean {}, sem {}".format(torch.mean(s_values_IG_trials, 0), sem(s_values_IG_trials, axis=0)))
#     print("SPGP IG-based MC-Shapley values: mean {}, sem {}".format(torch.mean(mc_s_values_IG_trials, 0), sem(mc_s_values_IG_trials, axis=0)))
#     print('-------------------------------------')
#
#     res['spgp_ig_sv'], res['spgp_ig_mc_sv'] = torch.mean(s_values_IG_trials, 0), torch.mean(mc_s_values_IG_trials, 0)


# suffix = '_rep' if rep else '' + '_superset' if superset else '' + '_train_test_diff_distri' if train_test_diff_distr else '' + '_size' if size else '' + '_disjoint' if disjoint else ''
suffix="differrent_task"
np.savez('../datasave/res_{}_{}D_{}M{}_repli_minmax.npz'.format("wine", D, M, suffix),
         res=res, M=M, D=D,  seed=seed)