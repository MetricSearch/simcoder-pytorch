import math
import numpy as np

def l1_norm(X):
    X = np.maximum(0,X)
    row_sums = np.sum(X,axis=1)
    X = np.divide(X.T,row_sums).T  # divide all elements rowwise by rowsums!
    return X

def complexity(X):
    """ returns a column vector with each  cell is the complexity of the rows of X"""
    # X is of shape no_datapoints,features
    X[X == 0] = 1                               # replace all the zeros with ones - get rid of log errors
    logs = np.log(X)                            # logs of shape no_of_objects,features
    hs = np.multiply(X,logs)                    # hs of shape no_of_objects,features print(f"hs {hs.shape}")
    cs = -np.nansum(hs, axis=1, keepdims=True)  # sum of x.logx along the rows => cs is of shape no_of_objects,1
    C = np.exp(cs)                              # C is of shape matrix of no_of_objects,1 print(f"C shape {C.shape}")
    return C                                    # return matrix of no_of_objects,1

class msed:
    """The data used with this must be l1_normed: use l1_norm function above."""

    def __init__(self,X):
        self.base_objects = X        
        self.no_of_objects = X.shape[0]                                 # a scalar - value is no_datapoint print(f"X.shape {X.shape}")
        self.base_sum = np.sum(X, axis=0, keepdims=True)                # column sums axis 0 so gives 1,encode_size print(f"base_sum.shape {self.base_sum.shape}")
        self.base_complexities = complexity(X)                          # num_of_objects,1 print(f"base_complexities.shape {self.base_complexities.shape}")      
        cp = np.cumprod(self.base_complexities)                         # num_of_objects, print(f"cp.shape {cp.shape}")                                        
        self.base_complexities_product = cp[self.no_of_objects-1];
        self.mean_values = np.sum(X, axis=0, keepdims=True) / self.no_of_objects  # mean column values - num_of_objects,1                 

    def msed(self,Y):  
        "returns a column vector of self.no_of_objects + 1,1 ????"
        # Y can be an array of data, each row of which is assessed against the pre-formed base
        no_of_vals = self.no_of_objects + 1;
        # M is also an array of data, each row is the overall mean - what does this mean?
        M = np.add(Y,np.sum(self.base_sum, axis=0)) / no_of_vals  # shape of M is 1,encode_size print(f"M shape {M.shape}") 
        top_line = complexity(M);                   # topLine is a column vector 1,1 print(f"top_line {top_line.shape}") 
        cprod = self.base_complexities_product;     # cprod is a float
        new_comp = complexity(Y);                   # new_comp is a column vector: encode_size,1 print(f"new_comp {new_comp.shape}")
        product = new_comp * cprod;                 # product is a column vector: encode_size,1 print(f"product {product.shape}") 
        bottom_line = product ** (1/no_of_vals)     # bottom_line is a column vector: encode_size,1 print(f"bottom_line {bottom_line.shape}") 
        result = (1 / self.no_of_objects) * (np.divide(top_line,bottom_line) - 1)
        # result is a column vector: Y.shape,1
        return result       
