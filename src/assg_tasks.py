import numpy as np
import math


def softmax(x):
    """Calculates the softmax for each row of the input 2-D tensor matrix x.
    Your code should work for a row vector of shape (1, n) but also for general
    matrices of shape (m, n).  You shouldn't have to do anything special to handl
    a row vector, numpy broadcasting should work in either case.

    Arguments
    ---------
    x - A numpy matrix (2-D tensor) of shape (m, n)

    Returns
    -------
    s - A numpy matrix with the same shape (m, n) as the input argument x, with the computed softmax of x
    """
    # apply the exponential function element-wise to x
    x_exp = np.exp(x)

    # create a vector that holds the sum of each row
    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    # compute softmax(x) by dividing x_exp by the row sums.  It should automatically use numpy broadcasting
    s = x_exp / x_sum

    return s

def sigmoid(x):
    """Compute sigmoid of the input parameter x and return. In this version
    the input parameter might be a scalar, but it could be a list or
    a numpy array.  Your implementation should be vectorized and able to
    hanld all of these.

    Arguments
    ---------
    x - a scalar, python list or numpy array of real valued (float/double) numbers.

    Returns
    -------
    s - Result will be of the same shape as the input and will be the element wise
      calculation of the sigmoid for all values given as input.
    """
    s = 1 / (1 + np.exp(-x))
    return s