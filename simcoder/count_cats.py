import numpy as np

from typing import Tuple

def get_topcat(index: int, sm_data: np.array) -> int:
    """Return the top category for the particular index"""
    data = sm_data[index]
    return np.argmax(data)

def findHighlyCategorisedInDataset(smData : np.array,thresh : float) -> Tuple[np.array, np.array]:
    """Find the most categorised data in the dataset
       To be included the images must be categorised at at least the theshold samenes
       Params: smData the softmax data for the dataset being analysed
            threshold - the value that any softmax value must reach to be included - higher is mre conservative
       Params: smData the softmax data for the dataset being analysed
            threshold - the value that any softmax value must reach to be included - higher is mre conservative
       Returns the most categorised categories (first) and their counts (second)"""
    filtered = np.where(smData>thresh,1,0) # 1 - s in all cells gt threshold
    sums = np.sum(filtered,axis=0) # sum up all the columns
    most_catgorical_indices = np.flip(np.argsort(sums))  # vector of indices sorted into descending order
    most_catgorical = sums[most_catgorical_indices] # sums in descending order
    return most_catgorical_indices,most_catgorical

def count_number_in_results_in_cat(cat,thresh,result_indices,sm_data):
    """Returns the number of results for which cat is greater than thresh
       Use the other function countNumberinCatGTThresh is just interested in encoding totals"""
    results = sm_data[result_indices]
    return ( results[:,cat] > thresh ).sum()

def getBestCats(category_required: int, sm_data: np.array) -> np.array:
    """Return the indices of those images sorted by the category_required in the sm_data"""
    indices = sm_data[:,category_required] # activation value for the category of interest for each nn
    best_nnids = np.flip(np.argsort(indices))
    return sm_data[best_nnids]

def get_best_cat_index(category_required: int, sm_data: np.array) -> np.array:
    """Return the indices of those images sorted by the category_required in the sm_data"""
    indices = sm_data[:,category_required] # activation value for the category of interest for each nn
    best_nnids = np.flip(np.argsort(indices))
    return best_nnids

def getBestCatsInSubset(category_required: int, subset: np.array, sm_data: np.array) -> np.array:
    """Return the indices of those images in nn_ids sorted by the category_required in the sm_data"""
    sm_values_of_interest = sm_data[subset]   # the sm values for each of the nns
    indices = sm_values_of_interest[:,category_required] # activation value for the category of interest for each nn
    best_nnids = np.flip(np.argsort(indices))
    return subset[best_nnids]

def countNumberinCatGTThresh(category: int,thresh: float,sm_data: np.array) -> int:
    """Returns the number of entries in encodings for which cat is greater than threshold
       Use the other function countNumberInResultsInCat if counting results"""
    return (sm_data[:,category] > thresh).sum()

def findCatsWithCountMoreThan(n: int,smData: np.array,thresh: float) -> np.array:
    """Finds the categories with more than n instances of a category whose value is nore than threshold"""
    vals = np.max(smData,axis=1) # 0 is rows, vals is the highest value in each row
    highest_labels_per_row = np.argmax(smData,axis=1) # the index of the highest value in each row
    highly_categorised_indices = highest_labels_per_row[vals>thresh] # the class indices of the images with category > thresh
    # now get rid of those that are less than n
    unique, counts = np.unique(highly_categorised_indices, return_counts=True) # get the unique category indices and their counts
    filtered = unique[counts>n] 
    return filtered

def findCatsWithCountMoreThanLessThan(n: int,k: int,smData: np.array,thresh: float) -> Tuple[np.array, np.array]:
    """Find the categories and their counts with more than n instances and less than k of a category whose value is nore than threshold"""
    vals = np.max(smData,axis=1) # 0 is rows, vals is the highest value in each row
    highest_labels_per_row = np.argmax(smData,axis=1) # the index of the highest value in each row
    highly_categorised_indices = highest_labels_per_row[vals > thresh] # the class indices of the images with category > thresh
    # now get rid of those that are less than n
    unique, counts = np.unique(highly_categorised_indices, return_counts=True) # get the unique category indices and their counts
    filtered = unique[(counts>n)&(counts<k)] 
    filtered_counts = counts[(counts>n)&(counts<k)]
    return filtered,filtered_counts

def findTopCats(smData: np.array,thresh: float) -> Tuple[np.array, np.array]:
    """An early version of findHighlyCategorisedInDataset - which is prbably better
       It finds the top categories in the dataset"""
    vals = np.max(smData,axis=1) # 0 is rows, vals is the highest value in each row

    column_totals = np.sum(smData,axis=0) # The totals of all the columns - what are the biggest categories?
    indices_of_most_categorical = np.argsort(column_totals)[::-1][:20] # reverse the and take the first 20

    highest_labels_per_row = np.argmax(smData,axis=1) # colum vect - the index of the highest value in each row
    data_above_thresh = smData[vals>thresh] # the class indices of the images with category > thresh

   # Need to count the entries that are in data_above_thresh and whose categories are in indices_of_most_categorical

    cats_in_most_categorical = np.isin(indices_of_most_categorical,highest_labels_per_row)
    top_cats_counts = smData[cats_in_most_categorical] # those rows in the most categorical
    top_cats_counts[vals>thresh] # those rows well categories

    return indices_of_most_categorical,top_cats_counts

def findHighCatValues(smData: np.array) -> np.array:
    """An early version of findHighlyCategorisedInDataset - which is prbably better"""
    """Returns the top 40 categories and their counts"""
    highest_vals = np.max(smData,axis=1) # highest_vals is the highest value in each row
    highest_index_per_row = np.argmax(smData,axis=1) # the index of the highest value in each row
    indices = np.argsort(highest_vals) # sorted list of the images with the least categorical categories
    indices = np.flip(indices) # reverse the list - most categorical categories
    highest_cats = highest_index_per_row[indices] # the categories of the images in descending categorical order (most sure first)
    return indices[0:40], highest_cats[0:40]
