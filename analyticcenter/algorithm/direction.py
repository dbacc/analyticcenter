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

from .analyticcenter import AnalyticCenter


class DirectionAlgorithm(object):
    """ 
    Base Class for all directional algorithms
    """
    debug = True
    discrete_time = None
    maxiter = None

    def __init__(self, riccati, initializer, abs_tol=np.finfo(float).eps, delta_tol=0., maxiter=100,
                 save_intermediate=False):
        self.riccati = riccati
        self.system = riccati.system
        self.abs_tol = abs_tol
        self.delta_tol = delta_tol
        if self.maxiter is None:
            self.maxiter = maxiter
        self.condition = 1. #TODO compute only every few steps.
        self.save_intermediate = save_intermediate
        if initializer is None:
            self.initializer = None
        else:
            self.initializer = initializer(riccati, None)
        if self.save_intermediate:
            self.intermediate_X = []
            self.intermediate_det = []
        logger = logging.getLogger(self.__class__.__name__)
        self.steps_count = 0

    def _print_information(self, residual, determinant, X):
        if (self.debug or self.logger.level == 0 or self.name == "NewtonMDCT" or self.name == "Steepest Ascent") and (
                        self.steps_count < 100 or self.steps_count % 100 == 0):
            self.logger.info("Current step: {}\tResidual: {}\tDet: {}".format(self.steps_count, residual, determinant))
            self.logger.debug("Current objective value (det(H(X))): {}".format(determinant))
            self.logger.debug("Current X:\n{}".format(X))
        det = np.real(determinant)
        if det < 0:
            self.logger.critical("Something went wrong. Determinant ist negative. Aborting...")
            raise ValueError("Something went wrong. Determinant ist negative")

    def _directional_iterative_algorithm(self, direction_algorithm, X0=None, fixed_direction=None):
        """
        
        Base algorithm for all directional iterative algorithms. 
        
        Parameters
        ----------
        direction_algorithm :
            
        X0 : If X0 is set, this value will be considered as initial guess
             (Default value = None)
        fixed_direction : If not None, search will only be performed along a given and fixed direction
             (Default value = None)

        Returns
        -------

        (analyticcenter, success) : A tuple of an analyticcenter object and a boolean success flag
        """

        if X0 is None:
            X, success_init = self.initializer()
        else:
            X, success_init = (X0, True)
        self.logger.debug("Initial X:\n{}".format(X))
        delta_cum = 0 * X
        P, R, F, A_F, residual, determinant = self.riccati.characteristics(X)
        Delta_residual = float("inf")
        alpha = 1.
        while residual > 10. * self.abs_tol * self.condition  and Delta_residual > self.delta_tol and self.steps_count < self.maxiter:
            self._print_information(residual, determinant, X)
            if self.save_intermediate:
                self._save_intermediate(X, determinant)

            Delta_X = direction_algorithm(X, P, R, A_F, fixed_direction)
            delta_cum += Delta_X
            Delta_residual = linalg.norm(Delta_X)
            X = X + alpha * Delta_X
            self.logger.debug("Updating current X by Delta:_X:\n{}".format(Delta_X))
            P, R, F, A_F, residual, determinant = self.riccati.characteristics(X, fixed_direction)
            self.steps_count += 1
        self._print_information(residual, determinant, X)
        if self.save_intermediate:
            self._save_intermediate(X, determinant)

        HX = self.riccati._get_H_matrix(X)
        analyticcenter = AnalyticCenter(X, A_F, HX, algorithm=self.riccati, discrete_time=self.discrete_time,
                                        delta_cum=delta_cum)
        if residual <= 10. * self.abs_tol * self.condition or Delta_residual <= self.delta_tol:
            if Delta_residual <= self.delta_tol:
                self.logger.warning(
                    "Residual of Delta: {} is below tolerance {}".format(Delta_residual, self.delta_tol))
            return (analyticcenter, True)
        else:
            self.logger.error("Computation failed. Maximal number of steps reached.")
            return (analyticcenter, False)

    def _save_intermediate(self, X, determinant):
            self.intermediate_X.append(X)
            self.intermediate_det.append(determinant)

    def __call__(self, X0=None):
        """
        Wrapper function for directional algorithm that does all the logging.

        Parameters
        ----------

        X0 : If X0 is set, this value will be considered as initial guess
             (Default value = None)

        Returns
        -------

        (analyticcenter, success) : A tuple of an analyticcenter object and a boolean success flag
        """
        self.logger.info("Computing Analytic Center with {} approach".format(self.name))
        analyticcenter, success = self._directional_iterative_algorithm(direction_algorithm=self._get_direction, X0=X0)
        if success:
            self.logger.info("Computation of Analytic center with {} approach was successful".format(self.name))
            self.logger.debug("At the analytic center A_F is:\n{}\nwith eigenvalues: {}".format(analyticcenter.A_F,
                                                                                                linalg.eig(
                                                                                                    analyticcenter.A_F)[
                                                                                                    0]))
            return analyticcenter, success
        else:
            self.logger.critical("Computation of Analytic center was  not successful")
            return analyticcenter, success
