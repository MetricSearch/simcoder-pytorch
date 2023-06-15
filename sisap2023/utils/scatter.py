# Code to create 2D projections of data.

import math 
import matplotlib.pyplot as plt
import numpy as np
from sisap2023.metrics.euc import euc_scalar, euc

def convertTo2D( pivot1,pivot2,data_points):
    '''Turn distances between the two pivots to the data_points into x,y coordinates'''
    dists1 = euc( pivot1,data_points )
    dists2 = euc( pivot2,data_points )
    ipd = euc_scalar( pivot1, pivot2 )
    x_offsets_apex = np.abs(( np.square(dists1) - np.square(dists2) + ipd**2 ) / (2 * ipd))
    y_offsets_apex = np.sqrt( np.square(dists1) - np.square(x_offsets_apex) )
    return x_offsets_apex,y_offsets_apex


def make2Dscatter(pivot1,pivot2,data_to_plot):
    '''Create a 2D scatter plot using the two pivot and the supplied data'''
    xs,ys = convertTo2D(pivot1,pivot2,data_to_plot)

    plt.scatter(xs, ys, c="blue")

    ipd = euc_scalar( pivot1,pivot2 )

    pivot1_x = [0]
    pivot1_y = [0]

    plt.scatter(pivot1_x, pivot1_y, c="red")

    pivot2_x = [ipd]
    pivot2_y = [0]

    plt.scatter(pivot2_x, pivot2_y, c="green")

    return plt