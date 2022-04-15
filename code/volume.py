import copy
from math import ceil, floor
from collections import defaultdict, Counter

import torch
import numpy as np
from torch import stack, cat, zeros_like, pinverse
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn import metrics
import  math
def compute_volumes(datasets, d=1):
    d = datasets[0].shape[1]
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reshape(-1 ,d)

    X = np.concatenate(datasets, axis=0).reshape(-1, d)
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det( dataset.T @ dataset ) + 1e-8)

    volume_all = np.sqrt(np.linalg.det(X.T @ X) + 1e-8).round(3)
    return volumes, volume_all

def compute_log_volumes(datasets, d=1):
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reshape(-1 ,d)

    X = np.concatenate(datasets, axis=0).reshape(-1, d)
    log_volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        log_volumes[i] = np.linalg.slogdet(dataset.T @ dataset)[1]

    log_vol_all = np.linalg.slogdet(X.T @ X)[1]
    return log_volumes, log_vol_all

def compute_pinvs(datasets, d=1):
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reshape(-1 ,d)
    X = cat(datasets).reshape(-1, d)

    zero_padded_datasets = []
    pinvs = []

    count = 0 
    for i, dataset in enumerate(datasets):
        zero_padded_dataset = zeros_like(X)
        # fill the total set X with the rows of individual dataset
        for j, row in enumerate(dataset):
            zero_padded_dataset[j+count] = row
        count += len(dataset)

        zero_padded_datasets.append(zero_padded_dataset)        

        pinvs.append(pinverse(zero_padded_dataset))

    pinv = pinverse(X)

    return pinvs, pinv


def compute_X_tilde_and_counts(X, omega):
    """
    Compresses the original feature matrix X to  X_tilde with the specified omega.

    Returns:
       X_tilde: compressed np.ndarray
       cubes: a dictionary of cubes with the respective counts in each dcube
    """

    D = X.shape[1]

    # assert 0 < omega <= 1, "omega must be within range [0,1]."

    m = ceil(1.0 / omega) # number of intervals for each dimension

    cubes = Counter() # a dictionary to store the freqs
    # key: (1,1,..)  a d-dimensional tuple, each entry between [0, m-1]
    # value: counts

    Omega = defaultdict(list)
    # Omega = {}
    
    min_ds = torch.min(X, axis=0).values

    # a dictionary to store cubes of not full size
    for x in X:
        cube = []
        for d, xd in enumerate(x - min_ds):
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

    # X_tilde = stack(list(Omega.values()))

    return X_tilde, cubes




def compute_robust_volumes(X_tildes, dcube_collections):
        
    N = sum([len(X_tilde) for X_tilde in X_tildes])
    alpha = 1.0 / (10 * N) # it means we set beta = 10
    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))
    d = X_tildes[0].shape[1]
    if d>10:
        volumes, volume_all = compute_log_volumes(X_tildes, d=X_tildes[0].shape[1])
    else:
        volumes, volume_all = compute_volumes(X_tildes, d=X_tildes[0].shape[1])
    robust_volumes = np.zeros_like(volumes)
    for i, (volume, hypercubes) in enumerate(zip(volumes, dcube_collections)):
        rho_omega_prod = 1.0
        for cube_index, freq_count in hypercubes.items():
            
            # if freq_count == 1: continue # volume does not monotonically increase with omega
            # commenting this if will result in volume monotonically increasing with omega
            rho_omega = (1 - alpha**(freq_count + 1)) / (1 - alpha)

            rho_omega_prod *= rho_omega

        robust_volumes[i] = (volume * rho_omega_prod).round(3)
    return robust_volumes



def compute_log_robust_volumes(X_tildes, dcube_collections):
        
    N = sum([len(X_tilde) for X_tilde in X_tildes])
    alpha = 1.0 / (10 * N) # it means we set beta = 10
    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    log_volumess, log_volume_all = compute_log_volumes(X_tildes, d=X_tildes[0].shape[1])
    log_robust_volumes = np.zeros_like(log_volumess)
    for i, (log_volume, hypercubes) in enumerate(zip(log_volumess, dcube_collections)):
        rho_omega_prod = 1.0
        for cube_index, freq_count in hypercubes.items():
            
            # if freq_count == 1: continue # volume does not monotonically increase with omega
            # commenting this if will result in volume monotonically increasing with omega
            rho_omega = (1 - alpha**(freq_count + 1)) / (1 - alpha)

            rho_omega_prod += np.log(rho_omega)

        log_robust_volumes[i] = (log_volume + rho_omega_prod).round(3)
    return log_robust_volumes




def compute_bias(S1, S2, d=1):
    X = np.append(S1, S2)

    S1 = S1.reshape(-1, d)
    S2 = S2.reshape(-1, d)
    X = X.reshape(-1, d)

    # print("data matrices shapes:", S1.shape, S2.shape, X.shape)
    XI_S1 = np.zeros( (X.shape[0], X.shape[0]) )
    XI_S2 = np.zeros( (X.shape[0], X.shape[0]) )

    IS1 = np.append(np.ones(s), np.zeros(s))
    IS2 = np.append(np.zeros(s), np.ones(s))
    for i in range(X.shape[0]):
        XI_S1[i,i] = IS1[i]
        XI_S2[i,i] = IS2[i]

    XI_S1 = XI_S1 @ X
    XI_S2 = XI_S2 @ X

    S1_pinv, S2_pinv = np.linalg.pinv(XI_S1), np.linalg.pinv(XI_S2)
    X_pinv = np.linalg.pinv(X)
    return np.linalg.norm(S1_pinv - X_pinv), np.linalg.norm(S2_pinv - X_pinv)

def compute_loss(S1, S2, f, d, param=False):
    y1 = np.asarray([f(s1) for s1 in S1 ])
    y2 = np.asarray([f(s2) for s2 in S2 ])
    # y1 = f(S1)
    # y2 = f(S2)
    
    X = np.append(S1, S2)
    y = np.append(y1, y2).reshape(-1, 1)

    S1 = S1.reshape(-1, d)
    S2 = S2.reshape(-1, d)
    X = X.reshape(-1, d)

    XI_S1_pinv, XI_S2_pinv = np.linalg.pinv(S1), np.linalg.pinv(S2)

    w_S1 = XI_S1_pinv @ y1
    w_S2 = XI_S2_pinv @ y2

    X_pinv = np.linalg.pinv(X)
    w_X =  X_pinv @ y

    loss1 = np.linalg.norm( X @ w_S1 - y )
    loss2 = np.linalg.norm( X @ w_S2 - y )
    loss = np.linalg.norm( X @ w_X - y )
    if not param:
        return loss1/len(y), loss2/len(y), loss/len(y)
    else:
        return loss1/len(y), loss2/len(y), loss/len(y), w_S1, w_S2


def check_freq_count(dcube_collections):
    for collection in dcube_collections:
        for cube_index, freq_count in collection.items():
            if freq_count != 1:
                print('freq count != 1', freq_count)
                break
    return


"""
Robustness helper functions
"""

def replicate(X, c, mode='full'):
    """
    Arguments: X: np.ndarray matrix of n x d shape representing the feature
               c: replication factor, c > 1
               mode: ['full', 'random']
               'full': replicate the entire dataset for c times
               'random': pick any row randomly to replicate (c*n) times
            
    Returns:
        A repliacted dataset
            X_r np.ndarray matrix of (n*c) x d shape
    """
    assert c > 1, "Replication factor must be larger than 1."
    c = int(c)
    if mode == 'full':
        X_replicated  = np.repeat(X, repeats=(c-1), axis=0)
        X_r = np.vstack((X, X_replicated))

    elif mode == 'random':
        L = len(X)
        probs = [1.0/L] * L
        repeats = np.random.multinomial(L* (c-1), probs, size=1)[0]
        X_replicated = np.repeat(X, repeats=repeats, axis=0)
        X_r = np.vstack((X, X_replicated))

    return torch.from_numpy(X_r)

def replicate_perturb(X, c=3, sigma=0.1):

    """
    Arguments: X: np.ndarray matrix of n x d shape representing the feature
               c: replication factor, c > 1
               sigma: the variance of zero-mean noise in each dimension
            
    Returns:
        A repliacted dataset
            X_r np.ndarray matrix of (n*c) x d shape
    """

    assert c > 1, "Replication factor must be larger than 1."
    c = int(c)

    assert 0 < sigma < 1, "For a standardized/normalized feature space, have sigma in [0, 1]."
    
    L = len(X)
    probs = [1.0/L] * L
    repeats = np.random.multinomial(L *(c-1), probs, size=1)[0]
    X_perturbed = np.repeat(X, repeats=repeats, axis=0)
    X_perturbed += np.random.normal(loc=0, scale=sigma, size=X_perturbed.shape)
    X_r = np.vstack((X, X_perturbed))

    return torch.from_numpy(X_r)


if __name__ == '__main__':
    d = 10
    s = d * 10

    for t in range(5):
        X = torch.from_numpy(np.random.normal(0, 1, (s,d)))
        v, _ = compute_volumes([X], d=d)
        print(v)
        break

    for t in range(5):
        X = torch.from_numpy(np.random.normal(0, 0.1, (s,d)))
        v, _ = compute_volumes([X], d=d)
        print(v)
        break
        rvs = []
        for o in [0.01, 0.1, 0.2, 0.49, 0.5, 0.51]:
            X_tilde, cubes = compute_X_tilde_and_counts(X, omega=o)
            rv = compute_robust_volumes([X_tilde], [cubes])
            if v[0] > rv:
                print('break', v[0], rv[0], o)

    exit()
    for t in range(100):
        X = torch.from_numpy(np.random.uniform(0, 1, (s,d)))
        v, _ = compute_volumes([X],  d=d)
        rvs = []
        for o in [0.01, 0.1, 0.2, 0.4, 0.49, 0.5, 0.51]:
            X_tilde, cubes = compute_X_tilde_and_counts(X, omega=o)
            rv = compute_robust_volumes([X_tilde], [cubes])
            if v[0] > rv:
                print('break', v[0], rv[0], o)


def calcuDistance(data1, data2):
    '''
    计算两个模式样本之间的欧式距离
    :param data1:
    :param data2:
    :return:
    '''
    distance = 0
    for i in range(len(data1)):
        distance += pow((data1[i] - data2[i]), 2)
    return math.sqrt(distance)


def maxmin_distance_cluster(data, Theta):
    '''
    :param data: 输入样本数据,每行一个特征
    :param Theta:阈值，一般设置为0.5，阈值越小聚类中心越多
    :return:样本分类，聚类中心
    '''
    maxDistance = 0
    start = 0  # 初始选一个中心点
    index = start  # 相当于指针指示新中心点的位置
    k = 0  # 中心点计数，也即是类别

    dataNum = len(data)
    distance = np.zeros((dataNum,))
    minDistance = np.zeros((dataNum,))
    classes = np.zeros((dataNum,))
    centerIndex = [index]

    # 初始选择第一个为聚类中心点
    ptrCen = data[0]
    # 寻找第二个聚类中心，即与第一个聚类中心最大距离的样本点
    for i in range(dataNum):
        ptr1 = data[i]
        d = calcuDistance(ptr1, ptrCen)
        distance[i] = d
        classes[i] = k + 1
        if (maxDistance < d):
            maxDistance = d
            index = i  # 与第一个聚类中心距离最大的样本

    minDistance = distance.copy()
    maxVal = maxDistance
    while maxVal > (maxDistance * Theta):
        k = k + 1
        centerIndex += [index]  # 新的聚类中心
        for i in range(dataNum):
            ptr1 = data[i]
            ptrCen = data[centerIndex[k]]
            d = calcuDistance(ptr1, ptrCen)
            distance[i] = d
            # 按照当前最近临方式分类，哪个近就分哪个类别
            if minDistance[i] > distance[i]:
                minDistance[i] = distance[i]
                classes[i] = k + 1
        # 寻找minDistance中的最大距离，若maxVal > (maxDistance * Theta)，则说明存在下一个聚类中心
        index = np.argmax(minDistance)
        maxVal = minDistance[index]
    return classes,centerIndex

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300,centers=None):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers_=centers
        self.clf_=None
    def fit(self, data):
        if self.centers_==None:
            self.centers_ = {}
            for i in range(self.k_):
                self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)

            for c in self.clf_:
                if len(self.clf_[c])==0:
                    return
                clusters=torch.stack(self.clf_[c])
                self.centers_[c] = torch.mean(clusters, dim=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if torch.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index

from sklearn.preprocessing import StandardScaler
def compute_X_tilde_and_counts_cluster(X, k_cluseter):
    scaler = StandardScaler()
    X = torch.from_numpy(scaler.fit_transform(X))

    D = X.shape[1]
    dataNum=X.shape[0]
    k=0
    m=[]
    m_k = torch.mean(X, dim=0)
    clf_all={}
    m.append(m_k)
    k+=1

    max_c=False

    norm_x_y=np.zeros((dataNum,dataNum))
    for i in range(dataNum):
        for j in range(dataNum):
            norm_x_y[i][j]=torch.norm(X[i]-X[j])
    while k<k_cluseter and max_c==False:
        k += 1

        min_distance=[]
        for i in range(dataNum):
            distance=[]
            for j in range(len(m)):
                d = calcuDistance(X[i],m[j])
                distance.append(d)
            min_index=distance.index(min(distance))
            min_distance.append(distance[min_index])

        bn_all=[]
        for  i in range(dataNum):
            bn=0
            for j in range (dataNum):
                bn+=max(min_distance[j]-norm_x_y[i][j],0)
            bn_all.append(bn)
        notend=True

        while notend:
            choose_index=bn_all.index(max(bn_all))
            choose_x=X[choose_index]

            current_m= {}
            for c in range(k-1):
                current_m[c]=m[c]
            current_m[k-1] =choose_x
            kmeans = K_Means(
                k=k,
                centers=current_m
            )
            kmeans.fit(X)

            notend=False
            for c in kmeans.clf_:
                if  len(kmeans.clf_[c])==1:
                    bn_all[choose_index]=0
                    notend=True
                    break
            # if notend==True:
            #     if (np.array(bn_all) == 0).all():
            #         max_c=True
            #         notend=False
            #         break
            # if notend==False and max_c==False:
        # if notend == False and max_c == False:
        for i in range(k-1):
            m[i]=kmeans.centers_[i]

        m.append(kmeans.centers_[k-1])
        clf_all = copy.deepcopy(kmeans.clf_)

    X_tilde=np.array([m[i].numpy() for i in range(len(m))]).reshape(len(m),-1)
    cubes = Counter()
    for c in clf_all:
        cubes[c]=len(clf_all[c])
    return X_tilde,cubes,0,0
def compute_X_tilde_and_counts_cluster_origin(X, k_cluseter):
    D = X.shape[1]
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(X)
    # classes, centerIndex = maxmin_distance_cluster(X, 0.2)
    # X_tilde=np.array([X[i].numpy() for i in centerIndex]).reshape(len(centerIndex),-1)
    # dist = cdist(X_tilde, X_tilde, metric='euclidean')
    # classes=np.array(classes)
    # cubes = Counter()
    # for label in classes:
    #     cubes[label] += 1
    #
    #
    # score = 0
    # # score=metrics.calinski_harabasz_score(X,label)
    # return X_tilde, cubes, dist, score
    kmeans = KMeans(
        init="k-means++",
        n_clusters=int(k_cluseter),
        max_iter=300,
        # random_state=42
    )


    kmeans.fit(X)

    X_tilde=kmeans.cluster_centers_
    # _, idx = np.unique(X_tilde, return_index=True)
    # X_tilde=X_tilde[np.sort(idx)]
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

def compute_robust_volumes_mine(X_tildes1, dcube_collections1,s):
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

def compute_single_volume(S1, d=1, method="normal"):
    S1 = S1.reshape(-1, d)
    if d>S1.shape[0] and d>10:
        v1 = np.linalg.slogdet(S1 @ S1.T)[1]
    if d > S1.shape[0] and d<= 10:
        v1 = np.sqrt(np.linalg.det(S1 @ S1.T) + 1e-8)
    if d <= S1.shape[0] and d>10:
        v1 = np.linalg.slogdet(S1.T @ S1)[1]
    if d <= S1.shape[0] and d<=10:
        a=S1.T @ S1
        v1 = np.sqrt(np.linalg.det(S1.T @ S1) + 1e-8)
    return v1
