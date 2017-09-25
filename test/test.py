import pytest

from analyticcenter.algorithm import get_analytic_center_object
from analyticcenter.direction import NewtonDirectionMultipleDimensionsCT, NewtonDirectionMultipleDimensionsDT
from analyticcenter.exceptions import AnalyticCenterUnstable, AnalyticCenterNotPassive, AnalyticCenterRiccatiSolutionFailed, AnalyticCenterUncontrollable
from examples.rlc import sys as sysrlc
from test.test_examples.example1 import sys
from test.test_examples.example4 import sys as sysuncontrollable
from examples.cheby_filter import sys as syscheby


def test_unstable_ct():
    with pytest.raises(AnalyticCenterUnstable):
        alg = get_analytic_center_object(sys, discrete_time=False)
        direction_method = NewtonDirectionMultipleDimensionsCT()
        (X, success) = direction_method()

def test_unstable_dt():
    with pytest.raises(AnalyticCenterUnstable):
        alg = get_analytic_center_object(sys, discrete_time=True)
        direction_method = NewtonDirectionMultipleDimensionsDT()
        (X, success) = direction_method()

def test_uncontrollable_ct():
    with pytest.raises(AnalyticCenterUncontrollable):
        alg = get_analytic_center_object(sysuncontrollable, discrete_time=False)
        direction_method = NewtonDirectionMultipleDimensionsCT()
        (X, success) = direction_method()

def test_2():
    alg = get_analytic_center_object(sysrlc, discrete_time=False)
    direction_method = NewtonDirectionMultipleDimensionsCT()
    (X, success) = direction_method()
    assert success

def test_cheby_no_riccati_solution():
    with pytest.raises(AnalyticCenterRiccatiSolutionFailed):
        alg = get_analytic_center_object(syscheby, discrete_time=False)
        direction_method = NewtonDirectionMultipleDimensionsCT()
        (X, success) = direction_method()