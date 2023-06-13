import math
import numpy as np
from sisap2023.metrics.euc import euc


def relu(X):
    return np.maximum(0,X)

def l1_norm(X):
    row_sums = np.sum(X,axis=1)
    X = np.divide(X.T,row_sums).T  # divide all elements rowwise by rowsums!
    return X

def l2_norm(X):
    # This only works if a matrix is passed in fails for vectors of a single row - TODO ho w to fix?
    origin = np.zeros(X.shape[1])
    factor = euc(origin,X)
    X = np.divide(X.T,factor).T
    return X

def get_dists(query_index,allData):
    '''Return the distances from the query to allData'''
    '''Returns an array same dimension as allData of scalars'''
    mf_query_data = allData[query_index]
    distances = euc(mf_query_data, allData)
    return distances