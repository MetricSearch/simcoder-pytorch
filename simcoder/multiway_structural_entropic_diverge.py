import numpy as np


def complexity(X: np.array) -> np.array:
    """Computes the complexity of the rows of X.

    Args:
        X (np.array): an (n,m) shaped array where n is the number of points 
        and m is the number of elements in each point.

    Returns:
        np.array: A column vector of shape (n,1) where each element is
        the complexity of one of the corresponding input row.
    """
    log_X = np.log(X)
    X_log_X = X * log_X
    sum_X_log_X = np.sum(X_log_X, axis=1, keepdims=True)
    res = np.exp(-sum_X_log_X)
    return res


def nth_root(a: np.array, n: int):
    return np.power(a,(1/n))


def multiway_structural_entropic_divergence(X: np.array) -> float:
    """Computes the multiway structural entropic divergence of a set of
    probability vectors where the sum of a vectors components equals 1.

    Args:
        X (np.array): an (n,m) shaped array where n is the number of vectors 
        and m is the number of elements in each vector.

    Returns:
        float: The MSED value - should be in the scale 0-1
    """
    assert np.all(np.isclose(np.sum(X, axis=1), 1.0)), "All row vectors in X must sum to 1."
    assert len(X.shape) == 2, "X must be a 2d array." 

    num_vectors = X.shape[0]
        
    # compute the numerator
    # a scalar - the complexity of the average of the vectors
    sums_of_vectors = np.sum(X, axis=0, keepdims=True)
    average_vector = np.divide(sums_of_vectors, num_vectors)
    complexity_of_average_vector = complexity(average_vector).item()

    # compute the denominator
    # a scalar - nth root of the product of the vector complexities
    vector_complexities = complexity(X)
    product_of_vector_complexities = np.product(vector_complexities)
    nth_root_of_product_of_vector_complexities = nth_root(product_of_vector_complexities, num_vectors)

    # compute the unscaled result 1.0-n
    res = complexity_of_average_vector / nth_root_of_product_of_vector_complexities
    assert res >= 1, f"Ratio should be greater than or equal to one. Value is {res}."
    assert res <= num_vectors, f"Ratio must be less than or equal to the number of vectors. Value is {res}."
    
    # scale the result to 0.0-1.0
    res = (res - 1) / (num_vectors - 1)
    return res

if __name__ == '__main__':
    base_matrix = np.array([[0.2, 0.6, 0.2],[0.1, 0.7, 0.2],[0.2, 0.5, 0.3]])
    data = np.array([[0.2, 0.7, 0.1]])
    test_data = np.vstack((base_matrix, data))
    msed_res = multiway_structural_entropic_divergence(test_data)
    print(msed_res)
