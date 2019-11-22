##
## Copyright (c) 2017
##
## @author: Daniel Bankmann
## @company: Technische Universität Berlin
##
## This file is part of the python package analyticcenter
## (see https://gitlab.tu-berlin.de/PassivityRadius/analyticcenter/)
##
## License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
##
import pytest
import analyticcenter
from analyticcenter import get_algorithm_object
from analyticcenter import AnalyticCenterUnstable, AnalyticCenterNotPassive, \
    AnalyticCenterRiccatiSolutionFailed, AnalyticCenterUncontrollable

from analyticcenter.examples.rlc import sys as sysrlc
from analyticcenter.examples.example2 import sys as sysex2
from test.test_examples.example1 import sys
from test.test_examples.example4 import sys as sysuncontrollable
from analyticcenter.examples.cheby_filter import sys as syscheby


def test_unstable_ct():
    with pytest.raises(AnalyticCenterUnstable):
        alg = get_algorithm_object(sys, 'newton', discrete_time=False)
        (X, success) = alg()


def test_unstable_dt():
    with pytest.raises(AnalyticCenterUnstable):
        alg = get_algorithm_object(sys, 'newton', discrete_time=True)

        (X, success) = alg()


# def test_uncontrollable_ct():
    # with pytest.raises(AnalyticCenterUncontrollable):
        # alg = get_algorithm_object(sysrlc, 'newton', discrete_time=False)
        # (X, success) = alg()


def test_2():
    alg = get_algorithm_object(sysex2, 'newton', discrete_time=False, abs_tol=9e-1)
    (X, success) = alg()
    assert success


def test_cheby_no_riccati_solution():
    with pytest.raises(AnalyticCenterRiccatiSolutionFailed):
        alg = get_algorithm_object(syscheby, 'newton', discrete_time=False)
        (X, success) = alg()
