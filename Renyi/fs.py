'''
author: Hong Ziming
Date: 1970-01-01 08:00:00
LastEditTime: 2020-12-26 05:59:26
LastEditors: Hong Ziming
Description:  
'''

import transplant
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
import multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


def feature_rank(**k):
    return


def main():

    matlab = transplant.Matlab(jvm=False, desktop=False)
    # c = matlab.selfadd(np.array([1,2,3,4]),np.array([2,2,4,4]))
    
    feature_name = scipy.io.loadmat(
        './mat/feature_name.mat')['feature_name']
    feature_name = [feature_name[0, i][0] for i in range(feature_name.shape[-1])]
    feature_name = np.array(feature_name)[np.newaxis,:]
    features = scipy.io.loadmat(
        './mat/features_norm.mat')['features_norm'].astype(np.float64)
    labels = scipy.io.loadmat(
        './mat/labels.mat')['labels'].astype(np.float64)

    num_features = 3
    fold_num = 5

    # split data
    skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=1234)
    for train_idx, test_idx in skf.split(features, labels):
        features_fold = features[train_idx]
        labels_fold = labels[train_idx] + 1
        select_idx = matlab.select_features_Renyi_nosvm(features_fold, labels_fold, num_features)
        select_idx = (np.squeeze(select_idx) - 1).astype(np.int)
        print(feature_name[:, select_idx])
        
    return
    

if __name__ == "__main__":
    main()
