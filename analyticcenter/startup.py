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
from .algorithm.riccatioperator import logger, RiccatiOperatorDiscreteTime, RiccatiOperatorContinuousTime
from .algorithm.newton import NewtonDirectionMultipleDimensionsCT, NewtonDirectionMultipleDimensionsDT
from .algorithm.steepestascent import SteepestAscentDirectionCT, SteepestAscentDirectionDT
from .algorithm.initialization import InitialXCT, InitialXDT
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _newtonmethod(discrete_time):
    if discrete_time:
        return NewtonDirectionMultipleDimensionsDT
    else:
        return NewtonDirectionMultipleDimensionsCT


def _steepestascentmethod(discrete_time):
    if discrete_time:
        return SteepestAscentDirectionDT
    else:
        return SteepestAscentDirectionCT


def _intialmethod(discrete_time):
    if discrete_time:
        return InitialXDT
    else:
        return InitialXCT


def _riccatiobject(discrete_time, system):
    if discrete_time:
        return RiccatiOperatorDiscreteTime(system)
    else:
        return RiccatiOperatorContinuousTime(system)


def get_algorithm_object(system, method, abs_tol=np.finfo(float).eps, delta_tol=0., maxiter=100,
                         discrete_time=None,
                         save_intermediate=False):
    """
   Function that returns an algorithm object.
    Parameters
    ----------
    system : WeightedSystem object
    method : Either 'Newton' or 'SteepestAscent'
    abs_tol : Sets the absolute tolerance for stopping the iteration
    delta_tol : Sets the tolerance for the increment Delta for stopping the iteration
    maxiter : Sets the maximum number of iterations
    discrete_time: Boolean, defining whether to use continous or discrete time version

    Returns
    -------
    Object of subclass of directional algorithm corresponding to the method and type of system.
    """
    if discrete_time is None:
        discrete_time = False
        logger.warning("No system type given. Defaulting to continuous time.")

    initializer = _intialmethod(discrete_time)
    if method.lower() == 'newton':
        algorithm = _newtonmethod(discrete_time)
    elif method.lower() == 'steepestascent':
        algorithm = _steepestascentmethod(discrete_time)
    else:
        logger.critical("No valid algorithm method given.")
        raise ValueError("No valid algorithm method given.")
    riccati = _riccatiobject(discrete_time, system)
    algorithm_object = algorithm(riccati, initializer, abs_tol, delta_tol, maxiter, save_intermediate)
    return algorithm_object
