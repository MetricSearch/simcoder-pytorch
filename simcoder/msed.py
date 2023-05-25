import math
import numpy as np

# function
# result = msed(X)
# noOfVals = size(X,1);
# avdata = sum(X) ./ noOfVals;
# compav = complexity(avdata);
# comps = complexity(X);
# product = cumprod(comps);
# bottomLine = product(noOfVals) ^ (1/noOfVals);
# result = (1 / (noOfVals - 1)) .* (compav / bottomLine - 1);
# end

def msed(X):
    """
    Calculates the mean squared exponential deviation (MSED) of input data.

    Args:
        X (ndarray): Input data as a 2D numpy array, where each row represents a sample.

    Returns:
        float: The MSED value.
    """
    X = np.abs(X) # take the L1 norm
    no_of_vals = X.shape[0]    # this is the number of values (rows) pased in - OK
    print(f"no_of_vals {no_of_vals}")
    
    av_data = np.divide(np.sum(X, axis=1), no_of_vals)  # sum across the rows gives (no_of_vals,) - OK
    #av_data = np.expand_dims(av_data, axis=1) # makes this into shape (no_of_vals,1) can call complexity on it

    comp_av = complexity(av_data)           # shape was (no_of_vals,1)    
    comps = complexity(X)                   # shape comps was (no_of_vals,) 
    product = np.cumprod(comps)              # shape product was (no_of_vals,)
    bottom_line = product[-1] ** (1 / no_of_vals)  # last index of product raised to 1/ no_of_vals (exp???)
    
    # This result should be a scalar - a float - but how do we get it?
    result = np.multiply((1 / (no_of_vals - 1)),(comp_av / bottom_line - 1))
    
    return result

# function
# C = complexity(X)
# logs = log(X);
# hs = X .* logs;
# cs = -sum(hs,2);
# C = exp(cs);
# end
def complexity(X: np.array) -> np.array:  # is this right or should be it be a scalar?
    """
    Calculates the complexity of input data.

    Args:
        X (ndarray): Input data as a 2D numpy array, where each row represents a sample.

    Returns:
        ndarray: Array of complexity values for each sample. 1D array as a column of floats.

    """
    logs = np.log(X)
    hs = np.multiply(X, logs)
    cs = - np.sum(hs)           #  a scalar  - a float  - did have an axis parameter. axis=1
    result = math.exp(cs)       #  a scalar  - a float  - WAS NP OPERATION DAVID
    return result
