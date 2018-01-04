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
import logging

import ipdb
import numpy as np
from scipy import linalg

from .direction import DirectionAlgorithm
from .newton import NewtonDirectionOneDimensionCT, NewtonDirectionOneDimensionDT


class SteepestAscentDirection(DirectionAlgorithm):
    """Class for the computation of the next iterate with the method of steepest ascent."""
    name = "Steepest Ascent"
    maxiter = 100000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_direction(self, X, P, R, A_F, fixed_direction=None):
        Delta = self._get_Delta(X, P, R, A_F)
        self.newton_direction._init_algorithm()
        (ac, success) = self.newton_direction._directional_iterative_algorithm(direction_algorithm=self.newton_direction._get_direction, X0=X, fixed_direction=Delta)
        self.logger.info("stepsize chosen by line_search: {}".format(-ac.delta_cum[0, 0]/Delta[0,0]))
        if np.isclose(linalg.norm(ac.delta_cum), 0):
            return Delta
        else:
            return ac.delta_cum


class SteepestAscentDirectionCT(SteepestAscentDirection):
    discrete_time = False
    line_search_method = NewtonDirectionOneDimensionCT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newton_direction = self.line_search_method(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_Delta(self, X, P, R, A):
        Delta = np.asmatrix(linalg.solve(P, A.H))
        Delta = Delta + Delta.H
        return Delta/linalg.norm(Delta)


class SteepestAscentDirectionDT(SteepestAscentDirection):
    discrete_time = True
    line_search_method = NewtonDirectionOneDimensionDT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newton_direction = self.line_search_method(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_Delta(self, X, P, R, A):
        B = self.system.B
        Pinv = linalg.inv(P)
        Delta = A @ Pinv @ A.H - Pinv + B @ linalg.solve(R - B.H @ X @ B, B.H)
        return Delta
