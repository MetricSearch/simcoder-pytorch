from dataclasses import dataclass
import math

from typing import List, Tuple
from pathlib import Path
import multiprocessing as mp

import click
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from sisap2023.utils.distances import get_dists, l1_norm, l2_norm, relu
from sisap2023.utils.count_cats import (
    count_number_in_cat_gt_thresh,
    count_number_in_results_cated_as,
    find_cats_with_count_more_than_less_than,
    get_best_cats_in_subset,
    get_best_cat_index,
)
from sisap2023.utils.mirflickr import load_encodings
from sisap2023.metrics.euc import euc_scalar
from sisap2023.metrics.msedOO import msedOO
from sisap2023.metrics.msed import msed
from sisap2023.metrics.nsimplex import NSimplex, fromSimplexPoint
from sisap2023.metrics.jsd_dist import jsd_dist

# Global constants - all global so that they can be shared amongst parallel instances

queries = None
top_categories = None
data = None  # the resnet 50 encodings
sm_data = None  # the softmax data
threshold = None
nn_at_which_k = None  # num of records to compare in results
category_names = None  # The categorical strings
best_k_for_queries = None
num_poly_queries = 6

# Functions:


def get_nth_categorical_query(categories: np.array, sm_data: np.array, n: int) -> List[int]:
    """Return the nth categorical query in each of the supplied categories"""
    results = []
    for cat_required in categories:
        cats = get_best_cat_index(cat_required, sm_data)  # all the data in most categorical order (not efficient!)
        results.append(cats[n])  # just get the nth categorical one
    return results


def select_poly_query_images(idx: int) -> Tuple[np.array, np.array]:
    """Takes the k-nn for the categories query image,
        orders them based on that categories softmax activation,
        returns the first num_poly_queries.

    Args:
        idx (int): The index of the query

    Returns:
        Tuple[np.array, np.array]: The image embedding and indices for the query images.
    """
    category, best_k_for_one_query = top_categories[idx], best_k_for_queries[idx]

    # the closest indices in category order - most peacocky peacocks etc.
    best_k_categorical = get_best_cats_in_subset(category, best_k_for_one_query, sm_data)

    # These are the indices that might be chosen by a human
    poly_query_indexes = best_k_categorical[0:num_poly_queries]

    # return the data and the indices
    poly_query_data = data[poly_query_indexes]
    return poly_query_data, poly_query_indexes


def compute_results(idx: int, distances: np.array) -> tuple:
    query, category, best_k_for_one_query = queries[idx], top_categories[idx], best_k_for_queries[idx]
    closest_indices = np.argsort(distances)  # the closest images
    best_k_for_poly_indices = closest_indices[0:nn_at_which_k]

    encodings_for_best_k_single = sm_data[best_k_for_one_query]
    encodings_for_best_k_poly = sm_data[best_k_for_poly_indices]

    max_possible_in_cat = count_number_in_cat_gt_thresh(category, threshold, sm_data)

    return (
        query,
        max_possible_in_cat,
        category,
        category_names[category],
        count_number_in_results_cated_as(category, best_k_for_queries[idx], sm_data),
        count_number_in_results_cated_as(category, best_k_for_poly_indices, sm_data),
        np.sum(encodings_for_best_k_single[:, category]),
        np.sum(encodings_for_best_k_poly[:, category]),
    )


def run_cos(idx: int):
    """This runs an experiment finding the NNs using cosine distance"""
    query = queries[idx]
    normed_data = l2_norm(data)
    distances = get_dists(query, normed_data)  # cosine distance same order as l2 norm of data
    return compute_results(idx, distances)


def run_jsd(idx: int):
    """This runs an experiment finding the NNs using SED"""
    """Uses the msed implementation"""
    query = queries[idx]
    relued_data = relu(data)
    normed_data = l1_norm(relued_data)
    distances = jsd_dist(normed_data[query], normed_data)
    return compute_results(idx, distances)


def run_sed(idx: int):
    """This runs an experiment finding the NNs using SED"""
    """Uses the msed implementation"""
    query = queries[idx]
    distances = np.zeros(1000 * 1000)
    for j in range(1000 * 1000):
        distances[j] = msed(np.vstack((data[query], data[j])))
    return compute_results(idx, distances)


def run_mean_point(idx: int):
    """This runs an experiment like perfect point below but uses the means of the distances to other pivots as the apex distance"""
    poly_query_data, poly_query_indexes = select_poly_query_images(idx)

    # poly_query_distances is the distances from the queries to the all data
    poly_query_distances = np.zeros((num_poly_queries, 1000 * 1000))
    for j in range(num_poly_queries):
        poly_query_distances[j] = get_dists(poly_query_indexes[j], data)

    # next line from Italian documentation: README.md line 25
    # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
    inter_pivot_distances = squareform(pdist(poly_query_data, metric=euc_scalar))

    apex_distances = np.mean(inter_pivot_distances, axis=1)

    # Here we set the perfect point to be at the mean inter-pivot distance.
    # mean_ipd = np.mean(inter_pivot_distances)
    # apex_distances = np.full(num_poly_queries,mean_ipd)
    # was multipled by 1.1 in some versions!
    distances = fromSimplexPoint(poly_query_distances, inter_pivot_distances, apex_distances)

    return compute_results(idx, distances)


def run_perfect_point(idx: int):
    """This runs an experiment with the the apex distance based on a NN distance from a simplex point"""
    poly_query_data, poly_query_indexes = select_poly_query_images(idx)

    # poly_query_distances is the distances from the queries to the all data
    poly_query_distances = np.zeros((num_poly_queries, 1000 * 1000))
    for j in range(num_poly_queries):
        poly_query_distances[j] = get_dists(poly_query_indexes[j], data)

    # Here we will use some estimate of the nn distance to each query to construct a
    # new point in the nSimplex projection space formed by the poly query objects
    nnToUse = 10
    ten_nn_dists = np.zeros(num_poly_queries)
    for i in range(num_poly_queries):
        sortedDists = np.sort(poly_query_distances[i])
        ten_nn_dists[i] = sortedDists[nnToUse]

    # next line from Italian documentation: README.md line 25
    # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
    inter_pivot_distances = squareform(pdist(poly_query_data, metric=euc_scalar))
    # was multipled by 1.1 in some versions!
    distances = fromSimplexPoint(poly_query_distances, inter_pivot_distances, ten_nn_dists)

    return compute_results(idx, distances)


def run_average(idx: int):
    """This just uses the average distance to all points from the queries as the distance"""
    _, poly_query_indexes = select_poly_query_images(idx)

    poly_query_distances = np.zeros((num_poly_queries, 1000 * 1000))
    for j in range(num_poly_queries):
        poly_query_distances[j] = get_dists(poly_query_indexes[j], data)

    distances = np.sum(poly_query_distances, axis=0)
    return compute_results(idx, distances)


def run_simplex(idx: int):
    "This creates a simplex and calculates the simplex height for each of the other points and takes the best n to be the query solution"
    poly_query_data, poly_query_indexes = select_poly_query_images(idx)

    # poly_query_distances is the distances from the queries to the all data
    poly_query_distances = np.zeros((num_poly_queries, 1000 * 1000))
    for j in range(num_poly_queries):
        poly_query_distances[j] = get_dists(poly_query_indexes[j], data)

    # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
    inter_pivot_distances = squareform(pdist(poly_query_data, metric=euc_scalar))

    # Simplex Projection
    # First calculate the distances from the queries to all data as we will be needing them again
    nsimp = NSimplex()
    nsimp.build_base(inter_pivot_distances, False)

    # Next, find last coord from the simplex formed by 6 query points
    all_apexes = nsimp._get_apex(nsimp._base, np.transpose(poly_query_distances))
    altitudes = all_apexes[:, num_poly_queries - 1]  # the heights of the simplex - last coordinate

    return compute_results(idx, altitudes)


def run_msed(idx: int):
    "This runs msed for the queries plus the values from the dataset and takes the lowest."
    _, poly_query_indexes = select_poly_query_images(idx)

    relued = relu(data)
    normed_data = l1_norm(relued)
    poly_query_data = normed_data[poly_query_indexes]

    base = msedOO(np.array(poly_query_data))
    msed_results = base.msed(normed_data)
    msed_results = msed_results.flatten()

    return compute_results(idx, msed_results)


def run_experiment(the_func, experiment_name: str, output_path: str):
    "A wrapper to run the experiments - calls the_func and saves the results from a dataframe"

    assert len(queries) == top_categories.size, "Queries and top_categories must be the same size."

    num_of_experiments = top_categories.size
    print(num_of_experiments)
    max_cpus = mp.cpu_count()
    use_cpus = max_cpus // 2

    print(f"Running {experiment_name} on {use_cpus} cpus from max of {max_cpus}")

    with mp.Pool(use_cpus) as p:
        xs = range(0, num_of_experiments)
        tlist = p.map(the_func, xs)

    # tlist is a list of tuples each tuple is the result of one run
    # which look like this: query, count best_k_for_one_query, best_k_for_expt, sum best_k_one, sum best_k_expt
    # now get a tuple of lists:

    unzipped = tuple(list(x) for x in zip(*tlist))
    # unzipped is a list of lists
    # now add the results to a dataframe and return it

    results = {
        "query": unzipped[0],
        "no_in_cat": unzipped[1],
        "cat_index": unzipped[2],
        "cat_string": unzipped[3],
        "nns_at_k_single": unzipped[4],
        "nns_at_k_poly": unzipped[5],
        "best_single_sums": unzipped[6],
        "best_poly_sums": unzipped[7],
    }

    print(f"Finished running {experiment_name}")

    results_df = pd.DataFrame(results)

    print(results_df.describe())
    results_df.to_csv(Path(output_path) / f"{experiment_name}.csv")


def compute_best_k_for_queries(queries: List[int], k: int):
    def closest(query):
        dists = get_dists(query, data)
        closest_indices = np.argsort(dists)
        return closest_indices[0:k]

    return [closest(q) for q in queries]


def load_imagenet_class_labels() -> List[str]:
    with open("imagenet_classes.txt", "r") as f:
        category_names = [s.strip() for s in f.readlines()]
    return category_names


@click.command()
@click.argument("encodings", type=click.Path(exists=False))
@click.argument("softmax", type=click.Path(exists=False))
@click.argument("output_path", type=click.Path(exists=False))
@click.argument("number_of_categories_to_test", type=click.INT)
@click.argument("k", type=click.INT)
@click.argument("initial_query_index", type=click.INT)
@click.argument("thresh", type=click.FLOAT)
def experimentselected(
    encodings: str,
    softmax: str,
    output_path: str,
    number_of_categories_to_test: int,
    k: int,
    initial_query_index: int,
    thresh: float,
):
    # These are all globals so that they can be shared by the parallel instances

    global data
    global sm_data
    global nn_at_which_k
    global threshold
    global top_categories
    global queries
    global category_names
    global best_k_for_queries

    print("Running experimentselected.")
    print(f"encodings: {encodings}")
    print(f"softmax: {softmax}")
    print(f"output_path: {output_path}")
    print(f"number_of_categories_to_test: {number_of_categories_to_test}")
    print(f"k: {k}")
    print(f"initial_query_index: {initial_query_index}")
    print(f"thresh: {thresh}")

    # Initialisation of globals

    print(f"Loading {encodings} data encodings.")
    data = load_encodings(Path(encodings))  # load resnet 50 encodings

    print(f"Loading {softmax} softmax encodings.")
    sm_data = load_encodings(Path(softmax))  # load the softmax data

    category_names = load_imagenet_class_labels()
    nn_at_which_k = k
    threshold = thresh

    # at least 80 and at most 195 - 101 cats sm values for resnet_50
    print("Finding highly categorised categories.")
    top_categories, _ = find_cats_with_count_more_than_less_than(100, 184, sm_data, threshold)
    top_categories = top_categories[0:number_of_categories_to_test]  # subset the top categories

    # with open("selected_queries.txt", "r") as f:
    #    queries = [int(line.strip()) for line in f]

    # get one query in each categories
    print(f"Finding {initial_query_index} categorical query for each category.")
    queries = get_nth_categorical_query(top_categories, sm_data, initial_query_index)

    print("Finding k-nn for each query.")
    best_k_for_queries = compute_best_k_for_queries(queries, nn_at_which_k)

    # end of Initialisation of globals - not updated after here

    run_experiment(run_perfect_point, "perfect_point", output_path)
    run_experiment(run_mean_point, "mean_point", output_path)
    run_experiment(run_simplex, "simplex", output_path)
    run_experiment(run_average, "average", output_path)
    run_experiment(run_msed, "msed", output_path)
    run_experiment(run_cos, "cos", output_path)
    run_experiment(run_sed, "sed", output_path)
    run_experiment(run_jsd, "jsd", output_path)
