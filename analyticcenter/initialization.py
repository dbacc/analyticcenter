import logging

import control
import numpy as np
from scipy import linalg

from analyticcenter.direction import DirectionAlgorithm
from analyticcenter.newton import NewtonDirectionOneDimensionCT, NewtonDirectionOneDimensionDT
import ipdb


class InitialX(DirectionAlgorithm):
    X0 = None
    maxiter = 100
    save_intermediate = False

    def __init__(self):
        pass

    def __call__(self):
        if InitialX.X0 is None:
            self.logger.info('Computing initial X')

            X_plus = -self.riccati_solver(self.system.A, self.system.B, self.system.Q, self.system.R, self.system.S,
                                          np.identity(self.system.n))
            Am = -self.system.A
            Bm = self.system.B
            Sm = self.system.S
            Qm = -self.system.Q
            Rm = -self.system.R
            X_minus = -self.riccati_solver(Am, Bm, Qm, Rm, Sm,
                                           np.identity(self.system.n))

            if np.isclose(linalg.norm(X_plus - X_minus), 0):
                self.logger.critical("X_+ and X_- are (almost) identical: No interior!")
            self.logger.debug("Eigenvalues of X_plus: {}".format(linalg.eigh(X_plus)[0]))
            self.logger.debug(
                "Eigenvalues of H(X_plus): {}".format(linalg.eigh(self.algorithm._get_H_matrix(X_plus))[0]))
            self.logger.debug("Eigenvalues of X_minus: {}".format(linalg.eigh(X_minus)[0]))
            self.logger.debug(
                "Eigenvalues of H(X_minus): {}".format(linalg.eigh(self.algorithm._get_H_matrix(X_minus))[0]))
            if self.debug:
                self.algorithm._get_Hamiltonian()
            # ipdb.set_trace()
            newton_direction = self.newton_direction
            fixed_direction = X_minus @ linalg.sqrtm(linalg.solve(X_minus, X_plus))
            # fixed_direction = 0.5* (X_minus + X_plus)
            # ipdb.set_trace()
            self.logger.debug("Eigenvalues of X_init_guess: {}".format(linalg.eigh(fixed_direction)[0]))
            self.logger.debug(
                "Eigenvalues of H(X_init_guess): {}".format(
                    linalg.eigh(self.algorithm._get_H_matrix(fixed_direction))[0]))
            InitialX.X0 = fixed_direction  # We use negative definite notion of solutions for Riccati equation
            # ipdb.set_trace()
            self.logger.info("Improving Initial X with Newton approach")

            # ipdb.set_trace()
            analyticcenter_init, success = self.newton_direction._directional_iterative_algorithm(
                direction_algorithm=newton_direction._get_direction, fixed_direction=fixed_direction)

            Xinit = analyticcenter_init.X
            if not success:
                self.logger.critical("Computation of initial X failed.")
            else:
                self.logger.debug("Eigenvalues of X_init: {}".format(linalg.eigh(Xinit)[0]))
                self.logger.debug(
                    "Eigenvalues of H(X_init): {}".format(linalg.eigh(self.algorithm._get_H_matrix(Xinit))[0]))

        else:
            # self.logger.info("Initial X is already set")
            Xinit = self.X0
        return Xinit, True


class InitialXCT(InitialX):
    newton_direction = NewtonDirectionOneDimensionCT()
    riccati_solver = staticmethod(lambda *args: control.care(*args)[0])

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)


class InitialXDT(InitialX):
    newton_direction = NewtonDirectionOneDimensionDT()
    riccati_solver = staticmethod(lambda *args: control.dare(*args)[0])

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
