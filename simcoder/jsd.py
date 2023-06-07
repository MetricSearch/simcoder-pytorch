# Jensen Shannon Distance
import math
import numpy as np

def jsd(A,B):
    """first param is an row - an array, second is matrix of rows (values)
       returns a column vector"""
    ha = h(A)
    hb = h(B)
    hc = h(A+B)
    hacc = ( hb - hc ) + ha
    he = - np.divide( np.nansum( hacc,1,keepdims=True ), math.log(4) ) + 1
    rtn = np.sqrt( np.where(he<0,0,he) )   
    return rtn

def h(x):
    return np.multiply( -x,np.log(x) )

# test code
# p1 = np.array([ 0, 0.1, 0.9 ])
# p2 = np.array([[ 0.9, 0, 0.1 ], [0, 0.5, 0.5]])
# print(jsd(p1,p2))
