import logging

import ipdb
import numpy as np
from scipy import linalg

from analyticcenter.direction import DirectionAlgorithm
from analyticcenter.newton import NewtonDirectionOneDimensionCT, NewtonDirectionOneDimensionDT


class DiagonalDirection(DirectionAlgorithm):
    name = "Diagonal"
    maxiter = 10000

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_direction(self, X, P, R, A_F, fixed_direction=None):
        B = self.system.B
        R = self.system.R
        ipdb.set_trace()
        Delta = 0.5*(np.diag(np.diag(A_F + A_F.H)) ** 2 @ linalg.pinv(np.diag(np.diag(B @ linalg.solve(R, B.H)))))

        return Delta
