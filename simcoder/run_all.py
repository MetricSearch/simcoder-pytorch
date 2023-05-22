import logging
import pandas as pd

import perfect_point_harness as pph
import average_dist_harness as adh
import simplex_harness as sph
import mean_point_harness as mph
from count_cats import findHighlyCategorisedInDataset

import time

from pathlib import Path
from similarity import load_mf_encodings
from similarity import load_mf_softmax
# import simcoder.perfect_point_harness as pph # moved below to aid reload
# Load the data:

logging.basicConfig(level=logging.ERROR)


data_root = Path("/Volumes/Data/")

data = load_mf_encodings(data_root / "mf_resnet50") # load resnet 50 encodings
sm_data = load_mf_softmax(data_root / "mf_softmax") # load the softmax data

print("Loaded datasets")

start_time = time.time()

nn_at_which_k : int = 100
number_of_categories_to_test : int = 100
threshold = 0.95

top_categories,counts = findHighlyCategorisedInDataset(sm_data, threshold)  # get the top categories in the dataset
top_categories = top_categories[0: number_of_categories_to_test]  # subset the top categories

queries = pph.getQueries(top_categories,sm_data)  # get one query in each category

print("Running perfect point.")
perp_results : pd.DataFrame = pph.run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k ) # TODO check the nn later
print(perp_results)
perp_results.to_csv(data_root / "perfect_point.csv")

print("Running simplex.")
simp_results : pd.DataFrame = sph.run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k ) # TODO check the nn later
print(simp_results)
simp_results.to_csv(data_root / "simplex.csv")

print("Running average.")
aver_results : pd.DataFrame = adh.run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k ) # TODO check the nn later
print(aver_results)
aver_results.to_csv(data_root / "average.csv")

mean_results : pd.DataFrame = mph.run_experiment(queries, top_categories, data, sm_data, threshold, nn_at_which_k ) # TODO check the nn later
print(mean_results)
mean_results.to_csv(data_root / "average.csv")

print("--- %s seconds ---" % (time.time() - start_time))

print(perp_results)

print(simp_results)

print(aver_results)


print(mean_results)
mean_results.to_csv(data_root / "average.csv")