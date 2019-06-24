##
## Copyright (c) 2019
##
## @author: Daniel Bankmann
## @company: Technische Universität Berlin
##
## This file is part of the python package analyticcenter
## (see https://gitlab.tu-berlin.de/PassivityRadius/analyticcenter/)
##
## License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
##
import numpy as np
from os.path import join, dirname
import analyticcenter
from analyticcenter import WeightedSystem, get_algorithm_object
from analyticcenter import NewtonDirectionMultipleDimensionsCT
from analyticcenter import SteepestAscentDirectionCT
sysmat = np.load(join(dirname(__file__), 'example-n-30-m-10.npy'), allow_pickle=True)
sys = WeightedSystem(*sysmat)
if __name__=="__main__":
    alg = get_algorithm_object(sys, 'newton', discrete_time=False, save_intermediate=True)
    alg.maxiter = 1000
    (ac_newton, success) = alg()

    sys_disc = sys.bilinear_discretization()

    alg_newton_disc = get_algorithm_object(sys_disc, 'newton', discrete_time=True, save_intermediate=True)
    (ac_newton_disc, success) = alg_newton_disc()


