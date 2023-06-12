from dataclasses import dataclass
import math

from typing import List
from pathlib import Path
import multiprocessing as mp

import click
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from sisap2023.utils.count_cats import countNumberinCatGTThresh

from sisap2023.utils.count_cats import (
    count_number_in_results_cated_as,
    findCatsWithCountMoreThanLessThan,
    getBestCatsInSubset,
    get_best_cat_index,
    count_number_in_results_in_cat,
    findHighlyCategorisedInDataset,
    get_topcat,
)
from sisap2023.utils.mirflickr import load_encodings
from sisap2023.utils.distances import euclid_scalar, get_dists, l1_norm, l2_norm, relu
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
categories = None  # The categorical strings

# Functions:


def get_nth_categorical_query(categories: np.array, sm_data: np.array, n: int) -> List[int]:
    """Return the nth categorical query in each of the supplied categories"""
    results = []
    for cat_required in categories:
        cats = get_best_cat_index(cat_required, sm_data)  # all the data in most categorical order (not efficient!)
        results.append(cats[n])  # just get the nth categorical one
    return results


def run_cos(i: int):
    """This runs an experiment finding the NNs using cosine distance"""
    query = queries[i]
    category = get_topcat(query, sm_data)

    assert get_topcat(query, sm_data) == category, "Queries and categories must match."

    dists = get_dists(query, data)
    closest_indices = np.argsort(dists)  # the closest images to the query
    best_k_for_one_query = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query

    normed_data = l2_norm(data)

    dists = get_dists(query, normed_data)  # cosine distance same order as l2 norm of data
    closest_indices = np.argsort(dists)  # the closest images to the query
    best_k_for_cosine = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query

    # Now want to report results the total count in the category

    encodings_for_best_k_single = sm_data[
        best_k_for_one_query
    ]  # the alexnet encodings for the best k average single query images
    encodings_for_best_k_cosine = sm_data[best_k_for_cosine]  # the alexnet encodings for the best cosine distances

    max_possible_in_cat = countNumberinCatGTThresh(category, threshold, sm_data)

    return (
        query,
        max_possible_in_cat,
        category,
        categories[category],
        count_number_in_results_cated_as(category, best_k_for_one_query, sm_data),
        count_number_in_results_cated_as(category, best_k_for_cosine, sm_data),
        np.sum(encodings_for_best_k_single[:, category]),
        np.sum(encodings_for_best_k_cosine[:, category]),
    )


def run_jsd(i: int):
    """This runs an experiment finding the NNs using SED"""
    """Uses the msed implementation"""

    query = queries[i]
    category = get_topcat(query, sm_data)
    dists = get_dists(query, data)
    closest_indices = np.argsort(dists)  # the closest images to the query

    best_k_for_one_query = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query

    relued_data = relu(data)
    normed_data = l1_norm(relued_data)

    jsd_results = jsd_dist(normed_data[query], normed_data)
    closest_indices = np.argsort(jsd_results)  # the closest images
    best_k_for_poly_indices = closest_indices[0:nn_at_which_k]

    # Now want to report results the total count in the category

    encodings_for_best_k_single = sm_data[
        best_k_for_one_query
    ]  # the alexnet encodings for the best k average single query images
    encodings_for_best_k_poly = sm_data[
        best_k_for_poly_indices
    ]  # the alexnet encodings for the best 100 poly-query images

    max_possible_in_cat = countNumberinCatGTThresh(category, threshold, sm_data)

    return (
        query,
        max_possible_in_cat,
        category,
        categories[category],
        count_number_in_results_cated_as(category, best_k_for_one_query, sm_data),
        count_number_in_results_cated_as(category, best_k_for_poly_indices, sm_data),
        np.sum(encodings_for_best_k_single[:, category]),
        np.sum(encodings_for_best_k_poly[:, category]),
    )


def run_sed(i: int):
    """This runs an experiment finding the NNs using SED"""
    """Uses the msed implementation"""

    query = queries[i]
    category = get_topcat(query, sm_data)
    dists = get_dists(query, data)
    closest_indices = np.argsort(dists)  # the closest images to the query

    best_k_for_one_query = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query

    sed_results = np.zeros(1000 * 1000)
    for j in range(1000 * 1000):
        sed_results[j] = msed(np.vstack((data[query], data[j])))

    closest_indices = np.argsort(sed_results)  # the closest images
    best_k_for_poly_indices = closest_indices[0:nn_at_which_k]

    # Now want to report results the total count in the category

    encodings_for_best_k_single = sm_data[
        best_k_for_one_query
    ]  # the alexnet encodings for the best k average single query images
    encodings_for_best_k_poly = sm_data[
        best_k_for_poly_indices
    ]  # the alexnet encodings for the best 100 poly-query images

    max_possible_in_cat = countNumberinCatGTThresh(category, threshold, sm_data)

    return (
        query,
        max_possible_in_cat,
        category,
        categories[category],
        count_number_in_results_cated_as(category, best_k_for_one_query, sm_data),
        count_number_in_results_cated_as(category, best_k_for_poly_indices, sm_data),
        np.sum(encodings_for_best_k_single[:, category]),
        np.sum(encodings_for_best_k_poly[:, category]),
    )


def run_mean_point(i: int):
    """This runs an experiment like perfect point below but uses the means of the distances to other pivots as the apex distance"""
    query = queries[i]
    category = get_topcat(query, sm_data)

    assert get_topcat(query, sm_data) == category, "Queries and categories must match."

    dists = get_dists(query, data)
    closest_indices = np.argsort(dists)  # the closest images to the query

    best_k_for_one_query = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query
    best_k_categorical = getBestCatsInSubset(
        category, best_k_for_one_query, sm_data
    )  # the closest indices in category order - most peacocky peacocks etc.
    poly_query_indexes = best_k_categorical[0:6]  # These are the indices that might be chosen by a human
    poly_query_data = data[poly_query_indexes]  # the actual datapoints for the queries
    num_poly_queries = len(poly_query_indexes)

    poly_query_distances = np.zeros(
        (num_poly_queries, 1000 * 1000)
    )  # poly_query_distances is the distances from the queries to the all data
    for j in range(num_poly_queries):
        poly_query_distances[j] = get_dists(poly_query_indexes[j], data)

    # next line from Italian documentation: README.md line 25
    inter_pivot_distances = squareform(
        pdist(poly_query_data, metric=euclid_scalar)
    )  # pivot-pivot distance matrix with shape (n_pivots, n_pivots)

    apex_distances = np.mean(inter_pivot_distances, axis=1)

    # Here we set the perfect point to be at the mean inter-pivot distance.
    # mean_ipd = np.mean(inter_pivot_distances)
    # apex_distances = np.full(num_poly_queries,mean_ipd)

    distsToPerf = fromSimplexPoint(
        poly_query_distances, inter_pivot_distances, apex_distances
    )  # was multipled by 1.1 in some versions!

    closest_indices = np.argsort(distsToPerf)  # the closest images to the perfect point
    best_k_for_poly_indices = closest_indices[0:nn_at_which_k]

    # Now want to report results the total count in the category

    encodings_for_best_k_single = sm_data[
        best_k_for_one_query
    ]  # the alexnet encodings for the best k average single query images
    encodings_for_best_k_poly = sm_data[
        best_k_for_poly_indices
    ]  # the alexnet encodings for the best 100 poly-query images

    max_possible_in_cat = countNumberinCatGTThresh(category, threshold, sm_data)

    return (
        query,
        max_possible_in_cat,
        category,
        categories[category],
        count_number_in_results_cated_as(category, best_k_for_one_query, sm_data),
        count_number_in_results_cated_as(category, best_k_for_poly_indices, sm_data),
        np.sum(encodings_for_best_k_single[:, category]),
        np.sum(encodings_for_best_k_poly[:, category]),
    )


def run_perfect_point(i: int):
    """This runs an experiment with the the apex distance based on a NN distance from a simplex point"""

    query = queries[i]
    category = get_topcat(query, sm_data)

    assert get_topcat(query, sm_data) == category, "Queries and categories must match."

    dists = get_dists(query, data)
    closest_indices = np.argsort(dists)  # the closest images to the query

    best_k_for_one_query = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query
    best_k_categorical = getBestCatsInSubset(
        category, best_k_for_one_query, sm_data
    )  # the closest indices in category order - most peacocky peacocks etc.
    poly_query_indexes = best_k_categorical[0:6]  # These are the indices that might be chosen by a human
    poly_query_data = data[poly_query_indexes]  # the actual datapoints for the queries
    num_poly_queries = len(poly_query_indexes)

    poly_query_distances = np.zeros(
        (num_poly_queries, 1000 * 1000)
    )  # poly_query_distances is the distances from the queries to the all data
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
    inter_pivot_distances = squareform(
        pdist(poly_query_data, metric=euclid_scalar)
    )  # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
    distsToPerf = fromSimplexPoint(
        poly_query_distances, inter_pivot_distances, ten_nn_dists
    )  # was multipled by 1.1 in some versions!

    closest_indices = np.argsort(distsToPerf)  # the closest images to the perfect point
    best_k_for_poly_indices = closest_indices[0:nn_at_which_k]

    # Now want to report results the total count in the category

    encodings_for_best_k_single = sm_data[
        best_k_for_one_query
    ]  # the alexnet encodings for the best k average single query images
    encodings_for_best_k_poly = sm_data[
        best_k_for_poly_indices
    ]  # the alexnet encodings for the best 100 poly-query images

    max_possible_in_cat = countNumberinCatGTThresh(category, threshold, sm_data)

    return (
        query,
        max_possible_in_cat,
        category,
        categories[category],
        count_number_in_results_cated_as(category, best_k_for_one_query, sm_data),
        count_number_in_results_cated_as(category, best_k_for_poly_indices, sm_data),
        np.sum(encodings_for_best_k_single[:, category]),
        np.sum(encodings_for_best_k_poly[:, category]),
    )


def run_average(i: int):
    """This just uses the average distance to all points from the queries as the distance"""

    query = queries[i]
    category = get_topcat(query, sm_data)
    dists = get_dists(query, data)
    closest_indices = np.argsort(dists)  # the closest images to the query

    best_k_for_one_query = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query
    best_k_categorical = getBestCatsInSubset(
        category, best_k_for_one_query, sm_data
    )  # the closest indices in category order - most peacocky peacocks etc.
    poly_query_indexes = best_k_categorical[0:6]  # These are the indices that might be chosen by a human
    poly_query_data = data[poly_query_indexes]  # the actual datapoints for the queries
    num_poly_queries = len(poly_query_indexes)

    poly_query_distances = np.zeros(
        (num_poly_queries, 1000 * 1000)
    )  # poly_query_distances is the distances from the queries to the all data
    for j in range(num_poly_queries):
        poly_query_distances[j] = get_dists(poly_query_indexes[j], data)

    row_sums = np.sum(poly_query_distances, axis=0)
    lowest_sum_indices = np.argsort(row_sums)

    best_k_for_poly_indices = lowest_sum_indices[:nn_at_which_k]

    # Now want to report results the total count in the category

    encodings_for_best_k_single = sm_data[
        best_k_for_one_query
    ]  # the alexnet encodings for the best k average single query images
    encodings_for_best_k_poly = sm_data[
        best_k_for_poly_indices
    ]  # the alexnet encodings for the best 100 poly-query images

    max_possible_in_cat = countNumberinCatGTThresh(category, threshold, sm_data)

    return (
        query,
        max_possible_in_cat,
        category,
        categories[category],
        count_number_in_results_cated_as(category, best_k_for_one_query, sm_data),
        count_number_in_results_cated_as(category, best_k_for_poly_indices, sm_data),
        np.sum(encodings_for_best_k_single[:, category]),
        np.sum(encodings_for_best_k_poly[:, category]),
    )


def run_simplex(i: int):
    "This creates a simplex and calculates the simplex height for each of the other points and takes the best n to be the query solution"

    query = queries[i]
    category = get_topcat(query, sm_data)
    dists = get_dists(query, data)
    closest_indices = np.argsort(dists)  # the closest images to the query

    best_k_for_one_query = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query
    best_k_categorical = getBestCatsInSubset(
        category, best_k_for_one_query, sm_data
    )  # the closest indices in category order - most peacocky peacocks etc.
    poly_query_indexes = best_k_categorical[0:6]  # These are the indices that might be chosen by a human
    poly_query_data = data[poly_query_indexes]  # the actual datapoints for the queries
    num_poly_queries = len(poly_query_indexes)

    poly_query_distances = np.zeros(
        (num_poly_queries, 1000 * 1000)
    )  # poly_query_distances is the distances from the queries to the all data
    for j in range(num_poly_queries):
        poly_query_distances[j] = get_dists(poly_query_indexes[j], data)

    inter_pivot_distances = squareform(
        pdist(poly_query_data, metric=euclid_scalar)
    )  # pivot-pivot distance matrix with shape (n_pivots, n_pivots)

    # Simplex Projection
    # First calculate the distances from the queries to all data as we will be needing them again

    nsimp = NSimplex()
    nsimp.build_base(inter_pivot_distances, False)

    # Next, find last coord from the simplex formed by 6 query points

    all_apexes = nsimp._get_apex(nsimp._base, np.transpose(poly_query_distances))
    altitudes = all_apexes[:, num_poly_queries - 1]  # the heights of the simplex - last coordinate

    closest_indices = np.argsort(altitudes)  # the closest images to the apex
    best_k_for_poly_indices = closest_indices[0:nn_at_which_k]

    # Now want to report results the total count in the category

    encodings_for_best_k_single = sm_data[
        best_k_for_one_query
    ]  # the alexnet encodings for the best k average single query images
    encodings_for_best_k_poly = sm_data[
        best_k_for_poly_indices
    ]  # the alexnet encodings for the best 100 poly-query images

    max_possible_in_cat = countNumberinCatGTThresh(category, threshold, sm_data)

    return (
        query,
        max_possible_in_cat,
        category,
        categories[category],
        count_number_in_results_cated_as(category, best_k_for_one_query, sm_data),
        count_number_in_results_cated_as(category, best_k_for_poly_indices, sm_data),
        np.sum(encodings_for_best_k_single[:, category]),
        np.sum(encodings_for_best_k_poly[:, category]),
    )


def run_msed(i: int):
    "This runs msed for the queries plus the values from the dataset and takes the lowest."

    relued = relu(data)
    normed_data = l1_norm(relued)

    query = queries[i]
    category = get_topcat(query, sm_data)
    dists = get_dists(query, normed_data)
    closest_indices = np.argsort(dists)  # the closest images to the query

    best_k_for_one_query = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query
    best_k_categorical = getBestCatsInSubset(
        category, best_k_for_one_query, sm_data
    )  # the closest indices in category order - most peacocky peacocks etc.
    poly_query_indexes = best_k_categorical[0:6]  # These are the indices that might be chosen by a human
    poly_query_data = normed_data[poly_query_indexes]  # the actual datapoints for the queries

    base = msedOO(np.array(poly_query_data))
    msed_results = base.msed(normed_data)
    msed_results = msed_results.flatten()

    closest_indices = np.argsort(msed_results)  # the closest images
    best_k_for_poly_indices = closest_indices[0:nn_at_which_k]

    # Now want to report results the total count in the category

    encodings_for_best_k_single = sm_data[
        best_k_for_one_query
    ]  # the alexnet encodings for the best k average single query images
    encodings_for_best_k_poly = sm_data[
        best_k_for_poly_indices
    ]  # the alexnet encodings for the best 100 poly-query images

    max_possible_in_cat = countNumberinCatGTThresh(category, threshold, sm_data)

    return (
        query,
        max_possible_in_cat,
        category,
        categories[category],
        count_number_in_results_cated_as(category, best_k_for_one_query, sm_data),
        count_number_in_results_cated_as(category, best_k_for_poly_indices, sm_data),
        np.sum(encodings_for_best_k_single[:, category]),
        np.sum(encodings_for_best_k_poly[:, category]),
    )


def run_experiment(the_func, experiment_name: str, output_path: str):
    "A wrapper to run the experiments - calls the_func and saves the results from a dataframe"

    assert len(queries) == top_categories.size, "Queries and top_categories must be the same size."
    assert len(queries) == top_categories.size, "Queries and top_categories must be the same size."

    num_of_experiments = top_categories.size

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

    print(results.describe())
    results.to_csv(Path(output_path) / f"{experiment_name}.csv")


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
    global categories

    print("Running experiment100.")
    print(f"encodings: {encodings}")
    print(f"softmax: {softmax}")
    print(f"output_path: {output_path}")
    print(f"initial_query_index: {initial_query_index}")

    # Initialisation of globals

    print(f"Loading {encodings} data encodings.")
    data = load_encodings(Path(encodings))  # load resnet 50 encodings

    print(f"Loading {softmax} softmax encodings.")
    sm_data = load_encodings(Path(softmax))  # load the softmax data

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    print("Loaded datasets")

    nn_at_which_k = k
    threshold = thresh

    print("Finding highly categorised categories.")
    top_categories, counts = findCatsWithCountMoreThanLessThan(
        100, 184, sm_data, threshold
    )  # at least 80 and at most 195 - 101 cats sm values for resnet_50
    top_categories = top_categories[0:number_of_categories_to_test]  # subset the top categories

    with open("selected_queries.txt", "r") as f:
        queries = [int(line.strip()) for line in f]

    queries = get_nth_categorical_query(
        top_categories, sm_data, initial_query_index
    )  # get one query in each categories

    # end of Initialisation of globals - not updated after here

    run_experiment(run_perfect_point, "perfect_point", output_path)
    run_experiment(run_mean_point, "mean_point", output_path)
    run_experiment(run_simplex, "simplex", output_path)
    run_experiment(run_average, "average", output_path)
    run_experiment(run_msed, "msed", output_path)
    run_experiment(run_cos, "cos", output_path)
    run_experiment(run_sed, "sed", output_path)
    run_experiment(run_jsd, "jsd", output_path)
