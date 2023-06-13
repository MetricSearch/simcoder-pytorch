import math
import numpy as np

def euc(img_features: np.array, encodings: np.array):
    distances = np.sqrt(np.sum(np.square((img_features - encodings)), axis=1))
    return distances

def euc_scalar(p1: np.array, p2: np.array):
    distances = math.sqrt(np.sum((p1 - p2) ** 2))
    return distances

