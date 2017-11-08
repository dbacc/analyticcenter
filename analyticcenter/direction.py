import logging
import ipdb
import numpy as np
from scipy import linalg

from .analyticcenter import AnalyticCenter


class DirectionAlgorithm(object):
    debug = True
    algorithm = None

    system = None
    initial_X = None
    discrete_time = False
    maxiter = 100
    fixed_direction = None
    save_intermediate = False
    intermediate_X = None
    intermediate_det = None
    name = None

    def __init__(self, ac_object, discrete_time, save_intermediate=False):
        DirectionAlgorithm.algorithm = ac_object
        DirectionAlgorithm.system = ac_object.system

        DirectionAlgorithm.discrete_time = discrete_time
        DirectionAlgorithm.save_intermediate = save_intermediate
        logger = logging.getLogger(self.__class__.__name__)

    def _print_information(self, steps_count, residual, determinant, X):
        self.logger.info("Current step: {}\tResidual: {}\tDet: {}".format(steps_count, residual, determinant))
        self.logger.debug("Current objective value (det(H(X))): {}".format(determinant))
        self.logger.debug("Current X:\n{}".format(X))
        det = np.real(determinant)
        if det < 0:
            self.logger.critical("Something went wrong. Determinant ist negative. Aborting...")
            raise ValueError("Something went wrong. Determinant ist negative")

    def _directional_iterative_algorithm(self, direction_algorithm, X0=None, fixed_direction=None):

        if X0 is None:
            X, success_init = self.initial_X()
        else:
            X, success_init = (X0, True)
        self.logger.debug("Initial X:\n{}".format(X))
        delta_cum = 0 * X
        F, P = self.algorithm._get_F_and_P(X)

        determinant = linalg.det(P) * self.algorithm._get_determinant_R(X)
        A_F = (self.system.A - self.system.B @ F)
        steps_count = 0
        residual = self.algorithm.get_residual(X, P, F, A_F, fixed_direction)
        Delta_residual = float("inf")
        # ipdb.set_trace()
        alpha = 1.
        # ipdb.set_trace()
        while residual > self.algorithm.abs_tol and Delta_residual > self.algorithm.rel_tol and steps_count < self.maxiter:
            # ipdb.set_trace()
            if self.debug:
                self.algorithm._get_H_matrix(X)
            # ipdb.set_trace()
            self._print_information(steps_count, residual, determinant, X)
            if self.save_intermediate:
                if self.intermediate_X is None:
                    self.intermediate_X = []
                    self.intermediate_det = []
                self.intermediate_X.append(X)
                self.intermediate_det.append(determinant)

            R = self.algorithm._get_R(X)
            self.logger.debug("Current Determinant of R: {}".format(linalg.det(R)))


            Delta_X = direction_algorithm(X, P, R, A_F, fixed_direction)
            if self.name == "NewtonMDCT" and steps_count>=20:
                ipdb.set_trace()
            #     Delta_X = linalg.solve(P, A_F.H)
            #     Delta_X += Delta_X.T
            delta_cum += Delta_X
            Delta_residual = linalg.norm(Delta_X)
            X = X + alpha * Delta_X
            self.logger.debug("Updating current X by Delta:_X:\n{}".format(Delta_X))

            F, P = self.algorithm._get_F_and_P(X)
            A_F = (self.system.A - self.system.B @ F)
            residual = self.algorithm.get_residual(X, P, F, A_F, fixed_direction)
            determinant = linalg.det(P) * self.algorithm._get_determinant_R(X)
            # ipdb.set_trace()
            steps_count += 1
        self._print_information(steps_count, residual, determinant, X)
        if self.save_intermediate:
            self.intermediate_X = np.array(self.intermediate_X)
            self.intermediate_det = np.array(self.intermediate_det)
        self.logger.info("Finished computation...")
        # if self.name == "NewtonMDCT":
        #     ipdb.set_trace()
        HX = self.algorithm._get_H_matrix(X)
        analyticcenter = AnalyticCenter(X, A_F, HX, algorithm=self.algorithm, discrete_time=self.discrete_time,
                                        delta_cum=delta_cum)
        if residual <= self.algorithm.abs_tol or Delta_residual <= self.algorithm.rel_tol:
            if Delta_residual <= self.algorithm.rel_tol:
                # ipdb.set_trace()
                self.logger.warning(
                    "Residual of Delta: {} is below tolerance {}".format(Delta_residual, self.algorithm.rel_tol))
            return (analyticcenter, True)
        else:
            self.logger.critical("Computation failed.")
            return (analyticcenter, False)

    def __call__(self, X0=None):
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
