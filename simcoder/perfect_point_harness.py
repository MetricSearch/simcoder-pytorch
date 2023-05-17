# Runs the perfect point experiment
# Assumes data is in /Volumes/Data
# 
# Returns a Pandas dataframe containing the  category_number, category_label, recall_at_k



# nasty import hack - this is a code smell, work out how to remove it
import sys
sys.path.append('../')


import pandas as pd
import numpy as np
import math
from simcoder.nsimplex import NSimplex
from simcoder.similarity import getDists
from simcoder.count_cats import getBestCats
from simcoder.count_cats import findHighlyCategorisedInDataset
from scipy.spatial.distance import pdist, squareform

from simcoder.count_cats import countNumberinCatGTThresh
from simcoder.count_cats import countNumberInResultsInCat

def euclid_scalar(p1: np.array, p2: np.array):
    distances = math.sqrt(np.sum((p1 - p2) ** 2))
    return distances

def fromSimplexPoint(poly_query_distances : np.array, inter_pivot_distances : np.array, nn_dists:  np.array) -> np.array:
    '''poly_query_data is the set of reference points with which to build the simplex'''
    '''inter_pivot_distances are the inter-pivot distances with which to build the base simplex'''
    '''nn_dists is a column vec of distances, each abit more than the nn distance from each ref to the rest of the data set'''
    '''ie the "perfect" intersection to the rest of the set'''
    '''returns a np.array of distances between the perfect point and the rest of the data set'''

    nsimp = NSimplex()
    nsimp.build_base(inter_pivot_distances,False)

    # second param a (B,N)-shaped array containing distances to the N pivots for B objects.
    perfPoint = nsimp._get_apex(nsimp._base,nn_dists)    # was projectWithDistances in matlab

    dists = np.zeros(1000 * 1000)
    for i in range(1000 * 1000):
        distvec = poly_query_distances[:,i];                      # a row vec of distances
        pr = nsimp._get_apex(nsimp._base,np.transpose(distvec));
        dists[i] = euclid_scalar(pr,perfPoint)  # is this right - see comment in simplex_peacock on this!

    return dists

def getQueries(): # TODO
    pass

def run_experiment_perfect_point(self, embeddings : np.array, categorised_embeddings : np.array, number_experiments : int, category_threshold : float, k : int ) -> pd.DataFrame:
    top_categories = findHighlyCategorisedInDataset(sm_data, threshold)  # get the top categories in the dataset
    top_categories = top_categories[:number_of_categories_to_test]  # subset the top categories
    queries = getQueries(top_categories)  # get one query in each category
    for i in range(top_categories):
        query = queries[i]
        category = top_categories[i]
        dists = getDists(query, data)
        closest_indices = np.argsort(dists)  # the closest images to the query
        best_k_for_one_query = closest_indices[0:nn_at_which_k]  # the k closest indices in data to the query
        best_k_categorical = getBestCats(category, best_k_for_one_query, sm_data,
                                         data)  # the closest indices in category order - most peacocky peacocks etc.
        poly_query_indexes = best_k_for_one_query[0:6]  # These are the indices that might be chosen by a human
        poly_query_data = data[poly_query_indexes]  # the actual datapoints for the queries
        num_poly_queries = len(poly_query_indexes)

        poly_query_distances = np.zeros(
            (num_poly_queries, 1000 * 1000))  # poly_query_distances is the distances from the queries to the all data
        for i in range(num_poly_queries):
            poly_query_distances[i] = getDists(poly_query_indexes[i], data)

            # Here we will use some estimate of the nn distance to each query to construct a
        # new point in the nSimplex projection space formed by the poly query objects

        nnToUse = 10
        ten_nn_dists = np.zeros(num_poly_queries);

        for i in range(num_poly_queries):
            sortedDists = np.sort(poly_query_distances[i]);
            ten_nn_dists[i] = sortedDists[nnToUse];

        # next line from Italian documentation: README.md line 25
        inter_pivot_distances = squareform(
            pdist(poly_query_data, metric=euclid_scalar))  # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
        distsToPerf = fromSimplexPoint(poly_query_distances, inter_pivot_distances,
                                       ten_nn_dists);  # was multipled by 1.1 in some versions!

        closest_indices = np.argsort(distsToPerf)  # the closest images to the perfect point
        best_100_for_perfect_point = closest_indices[0:100]

        # Now want to report results the total count in the category

        encodings_for_best_100_single = sm_data[
            best_k_for_one_query]  # the alexnet encodings for the best k average single query images
        best_single_totals = encodings_for_best_100_single[:, category]
        print("Total peacock sum for single query best 100: ", np.sum(best_single_totals))

        encodings_for_best_100_average = sm_data[
            best_100_for_perfect_point]  # the alexnet encodings for the best 100 polyquery images
        average_peacock_totals = encodings_for_best_100_average[:, category]
        print("Total peacock sum for poly query best 100: ", np.sum(average_peacock_totals))

        res = countNumberInResultsInCat(category, threshold, best_k_for_one_query, sm_data)
        print("Total with thresh better than threshold single query : ", res)

        res = countNumberInResultsInCat(category, 0.9, best_100_for_perfect_point, sm_data)
        print("Total with thresh better than threshold poly query : ", res)


