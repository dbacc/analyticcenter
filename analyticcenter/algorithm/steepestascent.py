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
    maxiter = 10000000

    def _get_stepsize(self, X, Delta):
        c1 = 10**4
        while True:
            P, R, F, A_F, residual, determinant = self.riccati.characteristics(X, None)
            Xn = X + self.stepsize * Delta
            Pn, Rn, Fn, A_Fn, residualn, determinantn = self.riccati.characteristics(Xn, None)
            if determinantn <=0:
                self.stepsize/= 2
                continue
            if np.log(determinantn) - np.log(determinant) >= self.stepsize / c1 * residual ** 2:
                self.stepsize *= 2
                return self.stepsize / 2
            else:
                self.stepsize /= 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stepsize = 1.

    def _get_direction(self, X, P, R, A_F, fixed_direction=None):
        Delta = self._get_Delta(X, P, R, A_F)
        stepsize = self._get_stepsize(X, Delta)
        self.logger.info("stepsize: {}".format(stepsize))
        return Delta * stepsize


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
        return Delta / linalg.norm(Delta)


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
