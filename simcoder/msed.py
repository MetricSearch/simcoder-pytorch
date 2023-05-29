import math
import numpy as np

def l1_norm(X):
    X = np.maximum(0,X)
    row_sums = np.sum(X,axis=1)
    X = np.divide(X.T,row_sums).T  # divide all elements rowwise by rowsums!
    return X
                 
def msed(X):
    X = l1_norm(X)                                                      # X is no_datapoints,features, Relud and L1 normed. - SHAPE CORRECT & DATA CORRECT
    noOfVals = X.shape[0]                                               # a scalar - value is no_datapoints
    avdata = np.sum(X, axis=0, keepdims=True) / noOfVals                # sum the columns find averages - shape: 1,features  - SHAPE CORRECT
    compav = complexity(avdata)                                         # the complexity of each of the rows (only 1) - so a single 1, vector
    comps = complexity(X)                                               # the complexity of each of the rows - datapoints,1
    product = np.cumprod(comps)                                         # product shape is no_datapoints,1
    bottomLine =  product[noOfVals-1] ** (1/noOfVals)                   # bottomline is a float
    result = (1 / (noOfVals - 1)) * (compav / bottomLine - 1)           # result is a 1,1 matrux need to unoack it.
    return result.item()        

def complexity(X):
    # X is of shape no_datapoints,features
    try:
        logs = np.log(X)                        # logs of shape no_datapoints,features
    except RuntimeWarning:                      # zero encountered - let it go (as per Disney)
        pass
    hs = np.multiply(X,logs)                    # hs of shape no_datapoints,features
    cs = -np.nansum(hs, axis=1, keepdims=True)  # sum along the rows => cs is of shape no_datapoints,1
    C = np.exp(cs)                              # C is of shape matrix of no_datapoints,1
    return C                                    # return matrix of no_datapoints,1