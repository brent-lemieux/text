import pytest

from inner_median import inner_median


def test_a():
    assert inner_median([3,1,2], [1,2,3,4]) == 2


def test_b():
    assert inner_median([1,3,2,1,3], [1,2,3,4]) == 2


def test_c():
    assert inner_median([1,1,3,2,1,3], [1,2,3,4]) == 2
