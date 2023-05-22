import functools
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd

import perfect_point_harness as pph
import average_dist_harness as adh
import simplex_harness as sph
import mean_point_harness as mph
from count_cats import findHighlyCategorisedInDataset
from perfect_point_harness import run_perfect_point

import time

from pathlib import Path
from similarity import load_mf_encodings
from similarity import load_mf_softmax
# import simcoder.perfect_point_harness as pph # moved below to aid reload
# Load the data:

def run_experiment(queries : np.array, top_categories: np.array, data: np.array, sm_data: np.array, threshold: float, nn_at_which_k: int, the_func ) -> pd.DataFrame:

    assert queries.size == top_categories.size, "Queries and top_categories must be the same size."    

    num_of_experiments = top_categories.size

    query_indices = np.zeros(num_of_experiments)  # used to collect results
    nns_at_k_single = np.zeros(num_of_experiments) # used to collect results
    nns_at_k_poly = np.zeros(num_of_experiments) # used to collect results
    best_single_sums = np.zeros(num_of_experiments) # used to collect results
    best_poly_sums = np.zeros(num_of_experiments) # used to collect results
    
    with Pool(num_of_experiments) as p:
        xs = range(0, num_of_experiments)
        res = p.map(functools.partial(the_func, queries=queries, top_categories=top_categories, data=data, sm_data=sm_data, threshold=threshold, nn_at_which_k=nn_at_which_k), xs)

    # now add the results to a dataframe and return it

    results = {
        "query": queries,
        "nns_at_k_single": nns_at_k_single,
        "nns_at_k_poly": nns_at_k_poly,
        "best_single_sums": best_single_sums,
        "best_poly_sums": best_poly_sums
    }

    return pd.DataFrame(results)

logging.basicConfig(level=logging.ERROR)


data_root = Path("/Volumes/Data/")

# encodings_name = 'mf_resnet50'
encodings_name = 'mf_dino2'
print(f"Loading {encodings_name} encodings.")
data = load_mf_encodings(data_root / encodings_name) # load resnet 50 encodings


print(f"Loading Alexnet Softmax encodings.")
sm_data = load_mf_softmax(data_root / "mf_softmax") # load the softmax data

print("Loaded datasets")

start_time = time.time()

nn_at_which_k : int = 100
number_of_categories_to_test : int = 2
threshold = 0.95

print("Finding highly categorised categories.")
top_categories,counts = findHighlyCategorisedInDataset(sm_data, threshold)  # get the top categories in the dataset
top_categories = top_categories[0: number_of_categories_to_test]  # subset the top categories

queries = pph.getQueries(top_categories,sm_data)  # get one query in each category

print("Running perfect point.")
perp_results : pd.DataFrame = pph.run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k,run_perfect_point ) # TODO check the nn later
print(perp_results)
perp_results.to_csv(data_root / f"perfect_point_{encodings_name}.csv")

# print("Running simplex.")
# simp_results : pd.DataFrame = sph.run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k ) # TODO check the nn later
# print(simp_results)
# simp_results.to_csv(data_root / f"simplex_{encodings_name}.csv")

# print("Running average.")
# aver_results : pd.DataFrame = adh.run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k ) # TODO check the nn later
# print(aver_results)
# aver_results.to_csv(data_root / f"average_{encodings_name}.csv")

# mean_results : pd.DataFrame = mph.run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k ) # TODO check the nn later
# print(mean_results)
# mean_results.to_csv(data_root / "average.csv")

print("--- %s seconds ---" % (time.time() - start_time))

print(perp_results.describe())

print(simp_results.describe())

print(aver_results.describe())

print(mean_results.describe())
