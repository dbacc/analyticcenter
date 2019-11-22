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
import numpy as np

from analyticcenter import WeightedSystem

sysmat = np.load('test/test_examples/example-n-6-m-1.npy', allow_pickle=True)
sys = WeightedSystem(*sysmat)
