"""This module contains the updated `inner_median` function.

To demonstrate that the function works as intended, run the command `py.test`.

To see a description of inefficiencies, read README.md Part One.
"""

import numpy as np


def inner_median(x, y):
    """Returns the median of the set intersection of two lists of integers.

    Ex: inner_median([1,2,3], [2,3,3]) would return 2.5 ([2,3] is the
    intersection of x, y and 2.5 is the median of [2,3]).

    Arguments:
        x (list<int>): A list of integers.
        y (list<int>): A list of integers.
    """
    intersection = np.intersect1d(x, y)
    return np.median(intersection)
