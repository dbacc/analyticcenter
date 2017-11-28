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
from analyticcenter import get_algorithm_object

from analyticcenter.examples.example2 import sys
import numpy as np

if __name__ == "__main__":

    alg = get_algorithm_object(sys, 'newton', discrete_time=False, save_intermediate=True)
    alg()

