#!/usr/bin/env python3


import logging
import numpy as np
from os.path import join, dirname
from analyticcenter import WeightedSystem, get_algorithm_object
from analyticcenter.visualize import log_log_direction

sysmat = np.load(join(dirname(__file__), 'example-n-6-m-3.npy'))
sys = WeightedSystem(*sysmat)

if __name__ == "__main__":
    logger = logging.getLogger(__file__)
    alg_newton = get_algorithm_object(sys, 'newton', discrete_time=False, save_intermediate=True)
    (ac_newton, success) = alg_newton()
    ac_newton.compute_characteristic_values()
    log_log_direction(alg_newton.intermediate_X, alg_newton.intermediate_det)
    logger.warning("The computation of the steepest ascent approach consumes a lot of time...")
    input('Press <ENTER> to continue')
    alg_steepest_ascent = get_algorithm_object(sys, 'steepestascent', discrete_time=False, save_intermediate=True)
    alg_steepest_ascent.maxiter = 1000000
    alg_steepest_ascent()
    log_log_direction(alg_steepest_ascent.intermediate_X, alg_steepest_ascent.intermediate_det)
