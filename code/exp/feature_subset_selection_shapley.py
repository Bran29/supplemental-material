### Robust Volume and Learning Performance Experiment
### Appendix B.4
import math
import random

import numpy as np
import torch
import copy

from itertools import permutations
from math import factorial

import sys
sys.path.insert(0, '..')
from main_utils import get_synthetic_datasets, generate_linear_labels, friedman_function, hartmann_function, scale_normal
from data_utils import load_uber_lyft, load_credit_card, load_hotel_reviews, load_used_car,load_breast_cancer,load_house_kc
from model import *
from volume import *
import argparse
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
# ---------- DATA PREPARATION ----------


def shapley_volume(Xs):
    M = len(Xs)
    S=Xs[0].shape[0]
    orderings = list(permutations(range(M)))
    # orderings=[[0],[1],[2],[3],[4]]
    s_values = torch.zeros(M)
    monte_carlo_s_values = torch.zeros(M)

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

            curr_train_X = torch.cat([dataset for j, dataset in enumerate(Xs) if j in curr_indices]).reshape(S,-1)

            curr_vol = torch.sqrt(torch.linalg.det(curr_train_X.T @ curr_train_X) + 1e-8)

            marginal = curr_vol - prefix_vol
            prefix_vol = curr_vol
            s_values[i] += marginal


            if ordering_count < K:
                monte_carlo_s_values[i] += marginal

    s_values /= factorial(M)


    print('------Volume-based Shapley value Statistics ------')
    print("Volume-based Shapley values:", s_values)
    print('-------------------------------------')
    return s_values

# def compute_losses_high_or_low_remove(feature_datasets, labels, feature_datasets_test, test_labels, M, omega, high_or_low,model,optimizer):
#
#     original_train_X = torch.cat(feature_datasets).reshape(-1, D)
#     original_train_y = torch.cat(labels).reshape(-1, 1)
#     test_X = torch.cat(feature_datasets_test).reshape(-1, D)
#     test_y = torch.cat(test_labels).reshape(-1, 1)
#
#     feature_datasets_, labels_, feature_datasets_test_, test_labels_ = copy.deepcopy(feature_datasets), copy.deepcopy(labels), copy.deepcopy(feature_datasets_test), copy.deepcopy(test_labels)
#
#     test_losses = []
#     train_losses = []
#
#     for i in range(M):
#         # Calculate current loss
#         train_X = torch.cat(feature_datasets_).reshape(-1, D)
#         train_y = torch.cat(labels_).reshape(-1, 1)
#
#         pinv = torch.pinverse(train_X)
#         test_loss = (torch.norm( test_X @ pinv @ train_y - test_y )) ** 2 / test_X.shape[0]
#         test_losses.append(test_loss.item())
#
#         train_loss = (torch.norm( original_train_X @ pinv @ train_y - original_train_y )) ** 2 / original_train_X.shape[0]
#         train_losses.append(train_loss.item())
#
#         # Calculate robust volumes
#         Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets_))
#         Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
#         robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)
#
#         if high_or_low == 'high':
#             idx = np.argmax(robust_volumes)
#         elif high_or_low == 'low':
#             idx = np.argmin(robust_volumes)
#         elif high_or_low == 'random':
#             idx = np.random.randint(low=0, high=robust_volumes.shape[0])
#         else:
#             raise NotImplementedError()
#
#         _ = feature_datasets_.pop(idx)
#         _ = labels_.pop(idx)
#
#     return test_losses, train_losses
def compute_losses_high_or_low_pv(feature_datasets, labels, feature_datasets_test, test_labels, M, omega, high_or_low,
                                   model, K_cluster, optimizer, args):
    original_train_X = torch.cat(feature_datasets).reshape(-1, args.D)
    original_train_y = torch.cat(labels).reshape(-1, 1)
    dataset = TensorDataset(original_train_X, original_train_y)  # create your datset
    all_loader = DataLoader(dataset, batch_size=50)  # create your dataloader

    test_X = torch.cat(feature_datasets_test).reshape(-1, args.D)
    test_y = torch.cat(test_labels).reshape(-1, 1)

    feature_datasets_, labels_, feature_datasets_test_, test_labels_ = [], [], [], []
    feature_datasets_copy, labels_copy, feature_datasets_test_copy, test_labels_copy = copy.deepcopy(
        feature_datasets), copy.deepcopy(labels), copy.deepcopy(feature_datasets_test), copy.deepcopy(test_labels)

    test_losses = []
    train_losses = []

    if args.method == "ref":
        Xtildes, dcube_collections = zip(
            *(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets_copy))
        Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
        robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)
    if args.method == "mine":
        robust_volumes = []
        Xtildes, cluster, dist, score = zip(
            *(compute_X_tilde_and_counts_cluster(dataset.numpy(), k_cluseter=K_cluster) for dataset in
              feature_datasets_copy))
        Xtildes, cluster_collections = list(Xtildes), list(cluster)
        for i in range(len(Xtildes)):
            robust_volumes.append(compute_robust_volumes_mine(Xtildes[i], cluster_collections[i], Xtildes[i].shape[0]))
        robust_volumes = np.array(robust_volumes)

    if high_or_low == 'high':
        idx = np.argmax(robust_volumes)
    elif high_or_low == 'low':
        idx = np.argmin(robust_volumes)
    elif high_or_low == 'random':
        idx = np.random.randint(low=0, high=robust_volumes.shape[0])
    else:
        raise NotImplementedError()

    dataset = TensorDataset(test_X, test_y)  # create your datset
    valid_loader = DataLoader(dataset, batch_size=1000)  # create your dataloader

    valid_loss = evaluate(model, valid_loader, args.loss_fn, args.device)

    # print(valid_loss)
    test_losses.append(valid_loss)
    for i in range(M):
        # feature_datasets_.append(feature_datasets_copy[idx])
        # labels_.append(labels_copy[idx])
        feature_datasets_ = [feature_datasets_copy[idx]]
        labels_ = [labels_copy[idx]]
        original_train_X = torch.cat(feature_datasets_).reshape(-1, args.D)
        original_train_y = torch.cat(labels_).reshape(-1, 1)
        train_batch_size = feature_datasets[0].shape[0]

        dataset = TensorDataset(original_train_X, original_train_y)  # create your datset
        loader = DataLoader(dataset, batch_size=feature_datasets[0].shape[0])  # create your dataloader
        if i == 0:
            train_loss = evaluate(model, all_loader, args.loss_fn, args.device)
            train_losses.append(train_loss)
        model = NN_Regressor(args.input_dim, args.output_dim, device=args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model, train_loss = train(model, loader, optimizer, args.loss_fn, args.E, args.device)
        valid_loss = evaluate(model, valid_loader, args.loss_fn, args.device)

        for para in model.parameters():
            print(para)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)

    # for i in range(M):
    #     Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets_copy))
    #     Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
    #     robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)
    #
    #     if high_or_low == 'high':
    #         idx = np.argmax(robust_volumes)
    #     elif high_or_low == 'low':
    #         idx = np.argmin(robust_volumes)
    #     elif high_or_low == 'random':
    #         idx = np.random.randint(low=0, high=robust_volumes.shape[0])
    #     else:
    #         raise NotImplementedError()
    #
    #     feature_datasets_.append(feature_datasets_copy.pop(idx))
    #     labels_.append(labels_copy.pop(idx))
    #
    #     # Calculate current loss
    #     train_X = torch.cat(feature_datasets_).reshape(-1, D)
    #     train_y = torch.cat(labels_).reshape(-1, 1)
    #
    #     pinv = torch.pinverse(train_X)
    #     test_loss = (torch.norm( test_X @ pinv @ train_y - test_y )) ** 2 / test_X.shape[0]
    #     test_losses.append(test_loss.item())
    #
    #     train_loss = (torch.norm( original_train_X @ pinv @ train_y - original_train_y )) ** 2 / original_train_X.shape[0]
    #     train_losses.append(train_loss.item())

    # return test_losses, train_losses
    return test_losses, train_losses



def compute_losses_high_or_low_add(feature_datasets, labels, feature_datasets_test, test_labels, M, omega, high_or_low,K_cluster,args):
    test_losses = []
    train_losses = []
    feature_datasets_copy, labels_copy, feature_datasets_test_copy, test_labels_copy = copy.deepcopy(
        feature_datasets), copy.deepcopy(labels), copy.deepcopy(feature_datasets_test), copy.deepcopy(test_labels)

    shapley_value = shapley_volume(feature_datasets_copy).numpy()

    if high_or_low == 'high':
        idx = np.argsort(shapley_value)[::-1]
    elif high_or_low == 'low':
        idx = np.argsort(shapley_value)
    elif high_or_low == 'random':
        idx = np.random.choice(shapley_value.shape[0], shapley_value.shape[0], replace=False)
    else:
        raise NotImplementedError()

    performance =[]
    for i in range(len(feature_datasets)):
        feature_datasets_copy=[]
        feature_datasets_test_copy=[]
        for j in range(i+1):
            feature_datasets_copy.append(feature_datasets[idx[j]])
            feature_datasets_test_copy.append(feature_datasets_test[idx[j]])
        # feature_datasets_test_copy = copy.deepcopy(feature_datasets_test)
        # feature_datasets_copy = copy.deepcopy(feature_datasets)
        # feature_datasets_test_copy= feature_datasets_test_copy.pop(idx[i])
        # feature_datasets_copy= feature_datasets_copy.pop(idx[i])
        # validition set
        dataset = TensorDataset(torch.hstack(feature_datasets_test_copy), test_labels_copy[0])
        valid_loader = DataLoader(dataset, batch_size=1000)
        #training set
        dataset = TensorDataset(torch.hstack(feature_datasets_copy), labels_copy[0])
        train_loader = DataLoader(dataset, batch_size=128)
        # initial model
        torch.manual_seed(args.seed)
        model = NN_Classification(len(feature_datasets_test_copy), args.output_dim, device=args.device)
        # model = NN_Regressor(len(feature_datasets_test_copy), args.output_dim, device=args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.5)
        # record parameter before training


        model,train_loss = train(model, train_loader, optimizer, args.loss_fn, args.E, args.device)
        valid_loss = evaluate(model, valid_loader, args.loss_fn, args.device)
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
    return test_losses, train_losses



def main():
    # config
    # 读取参数
    parser = argparse.ArgumentParser()
    # 实验参数

    device = torch.device('cpu')

    parser.add_argument('--function', default='breast', type=str, help='function')
    parser.add_argument('--exp_type', default='add', type=str)
    parser.add_argument('--method', default="mine", type=str)
    parser.add_argument('--n_participants', default=1, type=int)
    parser.add_argument('--s', default=800, type=int)
    parser.add_argument('--D', default=5, type=int)
    parser.add_argument('--omega', default=0.2, type=float)
    parser.add_argument('--cluster_num', default=20, type=int)
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--E', default=1000, type=int)
    parser.add_argument('--output_dim', default=1, type=int)

    args = parser.parse_args()
    function = args.function
    exp_type =  args.exp_type
    method =  args.method
    n_participants = M =  args.n_participants
    s =  args.s
    D =  args.D
    omega =  args.omega
    cluster_num =  args.cluster_num
    runs =  args.runs
    E =  args.E
    input_dim = args.D
    output_dim =  args.output_dim
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    args.loss_fn=loss_fn
    device = torch.device('cpu')
    args.device = device
    args.input_dim=input_dim

    high_test_arr = []
    high_train_arr = []
    low_test_arr = []
    low_train_arr = []
    rand_test_arr = []
    rand_train_arr = []

    high_pv_train_arr = []
    low_pv_train_arr = []
    rand_pv_train_arr = []

    for i in tqdm(range(runs)):
        seed = 1234 + i
        torch.manual_seed(seed)
        args.seed=seed
        if function == 'uber_lyft':
            assert D == 4
            feature_datasets, labels, feature_datasets_test, test_labels = load_uber_lyft(n_participants=M, s=s,
                                                                                          reduced=False,
                                                                                        path_prefix='../')
            device = torch.device('cpu')


        elif function == 'credit_card':
            assert D == 8
            feature_datasets, labels, feature_datasets_test, test_labels = load_credit_card(n_participants=M, s=s, path_prefix='../')
        elif function == 'hotel_reviews':
            assert D == 8
            feature_datasets, labels, feature_datasets_test, test_labels = load_hotel_reviews(n_participants=M, s=s, path_prefix='../')
        elif function == 'used_car':
            assert D == 5
            feature_datasets, labels, feature_datasets_test, test_labels = load_used_car(n_participants=M, s=s, train_test_diff_distr=False, path_prefix='../')
        elif function == 'breast':
            assert D == 5
            feature_datasets, labels, feature_datasets_test, test_labels = load_breast_cancer(n_participants=M, s=s, path_prefix='../')
        elif function == 'house_kc':
            assert D == 17
            feature_datasets, labels, feature_datasets_test, test_labels = load_house_kc(n_participants=M, s=s, path_prefix='../')
        else:
            raise NotImplementedError('Function not implemented.')

        feature_datasets, feature_datasets_test = scale_normal(feature_datasets, feature_datasets_test)
        # labels, test_labels = scale_normal(labels, test_labels)

        combines = [c for c in combinations(range(D), 1)]
        # combines = [(3), (16), (9), (2), (15)]

        # feature_datasets=[torch.cat((feature_datasets[0][:,c[0]].view(-1, 1),feature_datasets[0][:,c[1]].view(-1, 1)),1) for c in combines]
        # feature_datasets_test = [
        #     torch.cat((feature_datasets_test[0][:, c[0]].view(-1, 1), feature_datasets_test[0][:, c[1]].view(-1, 1)), 1) for c in
        #     combines]
        #
        feature_datasets = [feature_datasets[0][:, combines[i]].view(-1, 1) for i in range(len(combines))]
        feature_datasets_test = [feature_datasets_test[0][:, combines[i]].view(-1, 1) for i in range(len(combines))]

        labels = [labels[0] for i in range(len(combines))]
        test_labels = [test_labels[0] for i in range(len(combines))]

        if exp_type=='add':
            high_test, high_train = compute_losses_high_or_low_add(feature_datasets, labels, feature_datasets_test, test_labels, M, omega=0.1, high_or_low='high',K_cluster=cluster_num,args=args)
            low_test, low_train  = compute_losses_high_or_low_add(feature_datasets, labels, feature_datasets_test, test_labels, M, omega=0.1, high_or_low='low',K_cluster=cluster_num,args=args)
            rand_test, rand_train = compute_losses_high_or_low_add(feature_datasets, labels, feature_datasets_test, test_labels, M, omega=0.1, high_or_low='random',K_cluster=cluster_num,args=args)

        high_test_arr.append(high_test)
        high_train_arr.append(high_train)
        low_test_arr.append(low_test)
        low_train_arr.append(low_train)
        rand_test_arr.append(rand_test)
        rand_train_arr.append(rand_train)

    np.savez('../datasave/{}_valuable_{}_{}_{}_{}M_sv_performance_2l.npz'.format(exp_type, function,method,cluster_num,M),
             high_test=high_test_arr, high_train=high_train_arr, low_test=low_test_arr, low_train=low_train_arr, rand_test=rand_test_arr, rand_train=rand_train_arr,
             high_pv_train=high_pv_train_arr,low_pv_train=low_pv_train_arr,rand_pv_train=rand_pv_train_arr,M=M, D=D, s=s, function=function)

if __name__=="__main__":
    main()
