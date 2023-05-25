import numpy as np

def msed(X):
    """
    Calculates the mean structured entopic distance (MSED) of input data.

    Args:
        X (ndarray): Input data as a 2D numpy array, where each row represents a sample.

    Returns:
        float: The MSED value.

    """
    no_of_vals = X.shape[0]
    
    av_data = np.sum(X, axis=0) / no_of_vals
    comp_av = complexity(av_data)
    
    comps = complexity(X)
    product = np.cumprod(comps)
    bottom_line = product[no_of_vals - 1] ** (1 / no_of_vals)
    
    result = (1 / (no_of_vals - 1)) * (comp_av / bottom_line - 1)
    
    return result


def complexity(X):
    """
    Calculates the complexity of input data.

    Args:
        X (ndarray): Input data as a 2D numpy array, where each row represents a sample.

    Returns:
        ndarray: Array of complexity values for each sample.

    """
    logs = np.log(X)
    hs = X * logs
    cs = -np.sum(hs, axis=1)
    result = np.exp(cs)
    return result
