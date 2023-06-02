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
    X[X == 0] = 1  # replace all 0s with 1s to avoid log warnings
    log_X = np.log(X)
    X_log_X = X * log_X
    sum_X_log_X = np.nansum(X_log_X, axis=1, keepdims=True)
    res = np.exp(-sum_X_log_X)
    return res

def nth_root(a: np.array, n: int):
    """Computes the nth root of each element in the input array.

    Args:
        a (np.array): Input array.
        n (int): The root value.

    Returns:
        np.array: Resultant array with nth root of each element.
    """
    return np.power(a,(1/n))

class MSED:
    """Computes the multiway structural entropic divergence of a set of
    probability vectors where the sum of a vectors components equals 1.
    """
    def __init__(self, base: np.array) -> None:
        """Initialize the MSED instance.

        Args:
            base (np.array): The base vectors, a 2d array where each row represents a probability vector.
        """
        assert np.all(np.isclose(np.sum(base, axis=1), 1.0)), "All row vectors in the base vectors must sum to 1."
        assert len(base.shape) == 2, "The base vectors must be a 2d array." 

        self.num_vecs = base.shape[0]
        self.sums_of_vecs = np.sum(base, axis=0, keepdims=True)
        self.prod_of_vec_comp = np.product(complexity(base))

    def __call__(self, queries: np.array) -> np.array:
        """Compute the multiway structural entropic divergence for the given queries.

        Args:
            queries (np.array): The query vectors, a 2d array where each row represents a probability vector.

        Returns:
            np.array: The computed multiway structural entropic divergence for each query.
        """
        num_vecs = self.num_vecs + 1
        
        # compute the numerator
        # a scalar - the complexity of the average of the vectors for base + each query
        avg_vec = (queries + self.sums_of_vecs) / num_vecs
        comp_of_avg_vec = complexity(avg_vec)

        # compute the denominator
        # a scalar - nth root of the product of the vector complexities for base + each query
        comp_of_queries = complexity(queries)
        prod_of_vec_comp = self.prod_of_vec_comp * comp_of_queries
        nth_root_of_prod_of_vec_comp = nth_root(prod_of_vec_comp, num_vecs)

        # compute the unscaled result 1.0-n
        res = comp_of_avg_vec / nth_root_of_prod_of_vec_comp
        assert np.all(res >= 1), f"Ratio should be greater than 1."
        assert np.all(res <= num_vecs), f"Ratio must be less than or equal to number of vectors."

        # scale the result to 0.0-1.0
        res = (res - 1) / (num_vecs - 1)
        return res
    
if __name__ == '__main__':
    base = np.array([[0.2, 0.6, 0.2],[0.1, 0.7, 0.2],[0.2, 0.5, 0.3]])
    data = np.array([[0.2, 0.5, 0.3],[0.2, 0.7, 0.1]])
    msed = MSED(base)
    res = msed(data)
    print(res)
