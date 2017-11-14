import logging

import ipdb
import numpy as np
from scipy import linalg

from .direction import DirectionAlgorithm
from .newton import NewtonDirectionOneDimensionCT, NewtonDirectionOneDimensionDT


class SteepestAscentDirection(DirectionAlgorithm):
    name = "Steepest Ascent"
    newton = None
    maxiter = 10000

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_direction(self, X, P, R, A_F, fixed_direction=None):
        Delta = self._get_Delta(X, P, R, A_F)
        # ipdb.set_trace()
        (ac, success) = self.newton._directional_iterative_algorithm(direction_algorithm=self.newton._get_direction, X0=X, fixed_direction=Delta)

        return ac.delta_cum


class SteepestAscentDirectionCT(SteepestAscentDirection):
    newton = NewtonDirectionOneDimensionCT()

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_Delta(self, X, P, R, A):
        Delta = A + A.H
        return Delta/linalg.norm(Delta)


class SteepestAscentDirectionDT(SteepestAscentDirection):
    newton = NewtonDirectionOneDimensionDT()

    def _get_Delta(self, X, P, R, A):
        B = self.system.B
        Pinv = linalg.inv(P)
        Delta = A @ Pinv @ A.H - Pinv + B @ linalg.solve(R - B.H @ X @ B, B.H)
        return Delta/linalg.norm(Delta)