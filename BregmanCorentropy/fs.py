'''
author: Hong Ziming
Date: 2020-12-13 02:26:01
LastEditTime: 2020-12-17 23:02:52
LastEditors: Hong Ziming
Description:  
'''

import numexpr as ne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
from scipy.linalg import fractional_matrix_power
from scipy.linalg import logm
import scipy.io
from numpy.linalg import inv
from numpy.linalg import eig
from numpy import transpose as trans
import time
import os
import random
# from conditional_divergence import *
from BregmanCorentropy.conditional_divergence import *
import multiprocessing as mp
from tqdm import tqdm


def von_Neumann_Correntropy_eff(x1, y1, x2, y2, kernelSize):
    # compute positive semidefinite matrix
    C_x1_y1 = corrent_matrix(np.concatenate([x1, y1], axis=1), kernelSize)
    C_x2_y2 = corrent_matrix(np.concatenate([x2, y2], axis=1), kernelSize)
    C_x1 = corrent_matrix(x1, kernelSize)
    C_x2 = corrent_matrix(x2, kernelSize)
    # compute conditional divergence
    joint_divergence1 = von_Neumann_divergence_Eff(C_x1_y1, C_x2_y2)
    joint_divergence2 = von_Neumann_divergence_Eff(C_x2_y2, C_x1_y1)
    x_divergence1 = von_Neumann_divergence_Eff(C_x1, C_x2)
    x_divergence2 = von_Neumann_divergence_Eff(C_x2, C_x1)
    conditional_divergence = 1 / 2 * \
        (joint_divergence1 + joint_divergence2 - x_divergence1 - x_divergence2)

    return conditional_divergence


def von_Neumann_Correntropy(x1, y1, x2, y2, kernelSize):
    # compute positive semidefinite matrix
    C_x1_y1 = corrent_matrix(np.concatenate([x1, y1], axis=1), kernelSize)
    C_x2_y2 = corrent_matrix(np.concatenate([x2, y2], axis=1), kernelSize)
    C_x1 = corrent_matrix(x1, kernelSize)
    C_x2 = corrent_matrix(x2, kernelSize)
    # compute conditional divergence
    joint_divergence1 = von_Neumann_divergence(C_x1_y1, C_x2_y2)
    joint_divergence2 = von_Neumann_divergence(C_x2_y2, C_x1_y1)
    x_divergence1 = von_Neumann_divergence(C_x1, C_x2)
    x_divergence2 = von_Neumann_divergence(C_x2, C_x1)
    conditional_divergence = 1 / 2 * \
        (joint_divergence1 + joint_divergence2 - x_divergence1 - x_divergence2)

    return conditional_divergence
    

def perm_condition_divergence(cur_f, rem_f, rem_fnum, cur_fnum, labels, kernel_size):
    pidx = random.sample(range(rem_fnum), cur_fnum)
    perm_f = rem_f[:, pidx]
    d = von_Neumann_Correntropy(cur_f, labels, perm_f, labels, kernel_size)
    return d


def feature_rank(features, feature_name, labels, kernel_size=1, select_num=10, perm_num=100, num_cores=20):
    print('\nkernel_size=%.3f\nselect_num=%d\nperm_num=%d' %
          (kernel_size, select_num, perm_num))
    # set multiprocessing
    # num_cores = num_cores
    # num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)

    select_f = []
    select_fname = []

    for _ in range(select_num):
        print('[rank %03d]' % (len(select_f) + 1))
        divergence_all = np.zeros(features.shape[1])

        pbar = tqdm(range(features.shape[1]), unit='CandidateFeature')
        for remain_fidx in pbar:
            cur_f = np.concatenate(select_f + [features[:, [remain_fidx]]], axis=1)
            # cur_f = np.concatenate([select_f, features[:, [remain_fidx]]], axis=1)
            cur_fnum = cur_f.shape[1]

            rem_f = np.delete(features, remain_fidx, axis=1)
            rem_fnum = rem_f.shape[1]
            
            # multiprocessing for permutation
            perm_divergence_pools = [pool.apply_async(perm_condition_divergence, args=(
                cur_f, rem_f, rem_fnum, cur_fnum, labels, kernel_size)) for _ in range(perm_num)]
            perm_divergence = [p_pool.get() for p_pool in perm_divergence_pools]
            perm_divergence = np.array(perm_divergence)
            divergence = np.mean(perm_divergence)

            # perm_divergence = np.zeros(perm_num)
            # for p in range(perm_num):
            #     pidx = random.sample(range(rem_fnum), cur_fnum)
            #     perm_f = rem_f[:, pidx]
            #     d = von_Neumann_Correntropy(cur_f, labels, perm_f, labels, kernel_size)
            #     perm_divergence[p] = d
            # divergence = np.mean(perm_divergence)

            divergence_all[remain_fidx] = divergence

        select_fidx = np.argmax(divergence_all)
        select_f.append(features[:, [select_fidx]])
        # select_f = np.concatenate([select_f, features[:, [remain_fidx]]], axis=1)
        select_fname.append(feature_name[0, select_fidx])
        # select_fname.append(feature_name[0, select_fidx][0])
        print(select_fname[-1])
        features = np.delete(features, select_fidx, axis=1)
        feature_name = np.delete(feature_name, select_fidx, axis=1)

    return select_fname

        
def main():
    
    feature_name = scipy.io.loadmat(
        'BregmanCorentropy/mat/feature_name.mat')['feature_name']
    feature_name = [feature_name[0, i][0] for i in range(feature_name.shape[-1])]
    feature_name = np.array(feature_name)[np.newaxis,:]
    features = scipy.io.loadmat(
        'BregmanCorentropy/mat/features_norm.mat')['features_norm'].astype(np.float64)
    labels = scipy.io.loadmat(
        'BregmanCorentropy/mat/labels.mat')['labels'].astype(np.float64)

    kernel_size_list = [1]
    select_num_list = [3]
    perm_num_list = [100]

    for kernel_size in kernel_size_list:
        for select_num in select_num_list:
            for perm_num in perm_num_list:
                feature_rank(features.copy(), feature_name.copy(),
                             labels, kernel_size, select_num, perm_num)
    

if __name__ == "__main__":
    main()
