import functools
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd

from count_cats import findHighlyCategorisedInDataset
from perfect_point_harness import run_perfect_point, getQueries
from simplex_harness import run_simplex
from mean_point_harness import run_mean_point
from average_dist_harness import run_average

import time

from pathlib import Path
from similarity import load_mf_encodings
from similarity import load_mf_softmax
# import simcoder.perfect_point_harness as pph # moved below to aid reload
# Load the data:

data_root = Path("/Volumes/Data/")

def run_experiment(queries : np.array, top_categories: np.array, data: np.array, sm_data: np.array, threshold: float, nn_at_which_k: int, the_func ) -> pd.DataFrame:

    assert queries.size == top_categories.size, "Queries and top_categories must be the same size."    

    num_of_experiments = top_categories.size
     
    with mp.Pool(mp.cpu_count()) as p:
        xs = range(0, num_of_experiments)
        tlist = p.map(functools.partial(the_func, queries=queries, top_categories=top_categories, data=data, sm_data=sm_data, threshold=threshold, nn_at_which_k=nn_at_which_k), xs)

    # tlist is a list of tuples each tuple is the result of one run
    # which look like this: query, count best_k_for_one_query, best_k_for_expt, sum best_k_one, sum best_k_expt
    # now get a tuple of lists:

    unzipped = tuple(list(x) for x in zip(*tlist))
    # unzipped is a list of lists
    # now add the results to a dataframe and return it

    results = {
        "query": unzipped[0],
        "nns_at_k_single": unzipped[1],
        "nns_at_k_poly": unzipped[2],
        "best_single_sums": unzipped[3],
        "best_poly_sums": unzipped[4]
    }

    return pd.DataFrame(results)

def saveData( results: pd.DataFrame, expt_name : str,encodings_name: str) -> None:
    print(results.describe())
    results.to_csv(data_root / "results" / f"{expt_name}_{encodings_name}.csv" )

def main():

    # encodings_name = 'mf_resnet50'
    encodings_name = 'mf_dino2'
    print(f"Loading {encodings_name} encodings.")
    data = load_mf_encodings(data_root / encodings_name) # load resnet 50 encodings

    print(f"Loading Alexnet Softmax encodings.")
    sm_data = load_mf_encodings(data_root / "mf_alexnet_softmax") # load the softmax data

    print("Loaded datasets")

    start_time = time.time()

    nn_at_which_k : int = 100
    number_of_categories_to_test : int = 100
    threshold = 0.95

    print("Finding highly categorised categories.")
    top_categories,counts = findHighlyCategorisedInDataset(sm_data, threshold)  # get the top categories in the dataset
    top_categories = top_categories[0: number_of_categories_to_test]  # subset the top categories

    queries = getQueries(top_categories,sm_data)  # get one query in each category

    perp_results = run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k,run_perfect_point )
    mean_results = run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k,run_mean_point )
    simp_results = run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k,run_simplex )
    aver_results = run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k,run_average )

    print("--- %s seconds ---" % (time.time() - start_time))

    saveData( perp_results,"perfect_point",encodings_name)
    saveData( mean_results,"mean_point",encodings_name)
    saveData( simp_results,"simplex",encodings_name)
    saveData( aver_results,"average",encodings_name)

if __name__ == "__main__":
    mp.set_start_method("fork")
    main()