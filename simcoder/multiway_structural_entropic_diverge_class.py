from typing import Any
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

class MSED:
    def __init__(self, base: np.array) -> None:
        assert np.all(np.isclose(np.sum(base, axis=1), 1.0)), "All row vectors in the base vectors must sum to 1."
        assert len(base.shape) == 2, "The base vectors must be a 2d array." 

        self.num_base_vectors = base.shape[0]
        self.sums_of_base_vectors = np.sum(base, axis=0, keepdims=True)
        self.product_of_base_vector_complexities = np.product(complexity(base))

    def __call__(self, queries: np.array) -> np.array:
        num_vectors = self.num_base_vectors + 1
        
        # compute the numerator
        # a scalar - the complexity of the average of the vectors for base + each query
        average_vector_for_each_query = (queries + self.sums_of_base_vectors) / num_vectors
        complexity_of_average_vector_for_each_query = complexity(average_vector_for_each_query)

        # compute the denominator
        # a scalar - nth root of the product of the vector complexities for base + each query
        complexity_of_queries = complexity(queries)
        product_of_vector_complexities_for_each_query = self.product_of_base_vector_complexities * complexity_of_queries
        nth_root_of_product_of_vector_complexities_for_each_query = nth_root(product_of_vector_complexities_for_each_query, num_vectors)

        # compute the unscaled result 1.0-n
        res = complexity_of_average_vector_for_each_query / nth_root_of_product_of_vector_complexities_for_each_query
        assert np.all(res >= 1), f"Ratio should be greater than or equal to one. Array is {res}."
        assert np.all(res <= num_vectors), f"Ratio must be less than or equal to the number of vectors. Array is {res}."

        # scale the result to 0.0-1.0
        res = (res - 1) / (num_vectors - 1)
        return res
    
if __name__ == '__main__':
    base_matrix = np.array([[0.2, 0.6, 0.2],[0.1, 0.7, 0.2],[0.2, 0.5, 0.3]])
    data = np.array([[0.2, 0.5, 0.3],[0.2, 0.7, 0.1]])
    msed = MSED(base_matrix)
    res = msed(data)
    print(res)