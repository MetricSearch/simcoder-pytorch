import numpy as np

def countNumberInResultsInCat(cat,thresh,result_indices,sm_data):
    '''Returns the number of results for which cat is greater than thresh'''
    '''Use the other function countNumberinCatGTThresh is just interested in encoding totals'''
    results = sm_data[result_indices]
    return ( results[:,cat] > thresh ).sum()

def countNumberinCatGTThresh(cat,thresh,sm_data):
    '''Returns the number of entries in encodings for which cat is greater than thresh'''
    '''Use the other function countNumberInResultsInCat if counting results'''
    return (sm_data[:,cat] > thresh).sum()

def findCatsWithCountMoreThan(n,smData,thresh):
    vals = np.max(smData,axis=1) # 0 is rows, vals is the highest value in each row
    highest_labels_per_row = np.argmax(smData,axis=1) # the index of the highest value in each row
    highly_categorised_indices = highest_labels_per_row[vals>thresh] # the class indices of the images with category > thresh
    # now get rid of those that are less than n
    unique, counts = np.unique(highly_categorised_indices, return_counts=True) # get the unique category indices and their counts
    filtered = unique[counts>n] 
    return filtered

def findCatsWithCountMoreThanLessThan(n,k,smData,thresh):
    vals = np.max(smData,axis=1) # 0 is rows, vals is the highest value in each row
    highest_labels_per_row = np.argmax(smData,axis=1) # the index of the highest value in each row
    highly_categorised_indices = highest_labels_per_row[vals > thresh] # the class indices of the images with category > thresh
    # now get rid of those that are less than n
    unique, counts = np.unique(highly_categorised_indices, return_counts=True) # get the unique category indices and their counts
    filtered = unique[(counts>n)&(counts<k)] 
    filtered_counts = counts[(counts>n)&(counts<k)]
    return filtered,filtered_counts

def findTopCats(smData,thresh):
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

def findHighCatValues(smData):
    highest_vals = np.max(smData,axis=1) # highest_vals is the highest value in each row
    highest_index_per_row = np.argmax(smData,axis=1) # the index of the highest value in each row
    indices = np.argsort(highest_vals) # sorted list of the images with the least categorical categories
    indices = np.flip(indices) # reverse the list - most categorical categories
    highest_cats = highest_index_per_row[indices] # the categories of the images in descending categorical order (most sure first)
    return indices[0:40], highest_cats[0:40]
