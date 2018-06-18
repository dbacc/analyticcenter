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

import control
import numpy as np
from scipy import linalg

from .direction import DirectionAlgorithm
from .newton import NewtonDirectionOneDimensionCT, NewtonDirectionOneDimensionDT


class InitialX(DirectionAlgorithm):
    """Computation of Initial solution that is strictly positive for the LMI"""
    maxiter = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        Computation of an initial strictly positive solution of the LMI.
        We first compute two different solutions of the Riccati equation (stabilizing and anti-stabilizing).
        Then we hope, that combining them in a clever way leads us to a point that is in the interior.
        Here we use the geometric mean of both solutions.

        Returns
        -------
        Xinit : Initial solution

        success : Boolean
        """
        self.logger.info('Computing initial X')

        X_plus = -np.asmatrix(self.riccati_solver(self.system.A, self.system.B, self.system.Q, self.system.R, self.system.S,
                                      np.identity(self.system.n)))
        Am = -self.system.A
        Bm = self.system.B
        Sm = self.system.S
        Qm = -self.system.Q
        Rm = -self.system.R
        X_minus = -np.asmatrix(self.riccati_solver(Am, Bm, Qm, Rm, Sm,
                                       np.identity(self.system.n)))


        if np.isclose(linalg.norm(X_plus - X_minus), 0):
            self.logger.critical("X_+ and X_- are (almost) identical: No interior!")
        self.logger.debug("Eigenvalues of X_plus: {}".format(linalg.eigh(X_plus)[0]))
        self.logger.debug(
            "Eigenvalues of H(X_plus): {}".format(linalg.eigh(self.riccati._get_H_matrix(X_plus))[0]))
        self.logger.debug("Eigenvalues of X_minus: {}".format(linalg.eigh(X_minus)[0]))
        self.logger.debug(
            "Eigenvalues of H(X_minus): {}".format(linalg.eigh(self.riccati._get_H_matrix(X_minus))[0]))
        if self.debug:
            self.riccati._get_Hamiltonian()
        X0 = X_minus @ linalg.sqrtm(linalg.solve(X_minus, X_plus))
        fixed_direction = X0

        self.logger.debug("Eigenvalues of X_init_guess: {}".format(linalg.eigh(fixed_direction)[0]))
        self.logger.debug(
            "Eigenvalues of H(X_init_guess): {}".format(
                linalg.eigh(self.riccati._get_H_matrix(fixed_direction))[0]))

        # self.logger.info("Improving Initial X with Newton approach")
        # import pytest; pytest.set_trace()
        # analyticcenter_init, success = self.newton_direction._directional_iterative_algorithm(
        #     direction_algorithm=self.newton_direction._get_direction, fixed_direction=X0, X0=X0)

        # Xinit = analyticcenter_init.X
        # if not success:
        #     self.logger.critical("Computation of initial X failed.")
        # else:
        #     self.logger.debug("Eigenvalues of X_init: {}".format(linalg.eigh(Xinit)[0]))
        #     self.logger.debug(
        #         "Eigenvalues of H(X_init): {}".format(linalg.eigh(self.riccati._get_H_matrix(Xinit))[0]))

        return X0, True


class InitialXCT(InitialX):
    line_search_method = NewtonDirectionOneDimensionCT
    riccati_solver = staticmethod(lambda *args: control.care(*args)[0])


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newton_direction = self.line_search_method(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)



class InitialXDT(InitialX):
    line_search_method = NewtonDirectionOneDimensionDT
    riccati_solver = staticmethod(lambda *args: control.dare(*args)[0])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newton_direction = self.line_search_method(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

