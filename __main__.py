import inspect
import logging
import os

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import linalg

from logger import prepare_logger


def rsolve(*args, **kwargs):
    return np.asmatrix(linalg.solve(np.asmatrix(args[0]).H, np.asmatrix(args[1]).H, kwargs)).H


def load_config():
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    configpath = os.path.join(cmd_folder, 'config/config.yaml')
    with open(configpath, 'rt') as f:
        config = yaml.safe_load(f.read())
    return config


class LTI(object):
    """Describes an LTI system"""

    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.p = C.shape[0]
        self.logger = logging.getLogger(__name__)


class OptimalControlSystem(LTI):
    """Describes LTI system with weight Matrix"""

    def __init__(self, A, B, C, D, Q, S, R):
        super().__init__(A, B, C, D)
        self.Q = Q
        self.S = S
        self.R = R
        self.__initH0()
        self._check_positivity_weight_matrix()

    def __initH0(self):
        self.H0 = np.bmat([[self.Q, self.S], [self.S.transpose(), self.R]])

    def _check_positivity_weight_matrix(self):
        try:
            linalg.cholesky(self.H0)
            self.logger.info('Positive definite')
        except linalg.LinAlgError as err:
            self.logger.info('Not positive definite')


class AnalyticCenter(object):
    """ToDo"""

    def __init__(self, system, tol):
        self.system = system
        self.H0 = None
        self.H = None
        self.tol = tol
        self.maxiter = 1000
        self.__init_H0()
        self.debug = True
        self.logger = logging.getLogger(__name__)

    def __init_H0(self):
        self.H0 = np.bmat([[self.system.Q, self.system.S], [self.system.S.transpose(), self.system.R]])

    def _get_H_matrix(self, X: np.matrix):
        A = self.system.A
        B = self.system.B
        H = self.H0 - np.bmat([[A.T @ X + X @ A, X @ B],
                               [B.T @ X, np.zeros((self.system.m, self.system.m))]])
        return H

    def _get_Delta_H(self, Delta_X):
        return self._get_H_matrix(Delta_X) - self.H0

    def _get_initial_X(self):
        self.logger.info('Computing initial X')
        X_plus = linalg.solve_continuous_are(self.system.A, self.system.B, self.system.Q, self.system.R)
        Am = -self.system.A
        Bm = -self.system.B
        X_minus = linalg.solve_continuous_are(Am, Bm, self.system.Q, self.system.R)
        return -0.5 * (X_minus + X_plus)  # We use negative definite notion of solutions for Riccati equation

    def _get_F_and_P(self, X):
        F = linalg.solve(self.system.R, self.system.S.H - self.system.B.H @ X)
        P = self.riccati_operator(X)
        return F, P

    def steepest_ascent(self):
        self.logger.info("Computing Analytic Center with steepest_ascent approach")
        determinant_R = linalg.det(self.system.R)
        # noinspection PyPep8Naming
        X = self._get_initial_X()
        self.logger.debug("Initial X:\n{}".format(X))
        F, P = self._get_F_and_P(X)
        determinant = linalg.det(P) * determinant_R
        A_F = (self.system.A - self.system.B @ F)
        steps_count = 0
        residual = self.get_residual(X, P, F, A_F)
        while residual > self.tol and steps_count < self.maxiter:
            steps_count += 1
            self.logger.info("Current step: {}\tResidual: {}".format(steps_count, residual))
            self.logger.debug("Current objective value (det(H(X))): {}".format(determinant))
            self.logger.debug("Current X:\n{}".format(X))
            self.logger.debug("Current P:\n{}".format(P))
            # Delta_X = self._get_ascent_direction(X, A_F)
            # maxindex = self.gradient_sweep(X)
            # Delta_X = 0 * X
            # Delta_X[maxindex] = 1
            # Delta_X = Delta_X.T + Delta_X
            # Delta_X *= self.det_direction_plot(X, Delta_X)
            # ipdb.set_trace()
            Delta_X = self._get_ascent_direction(X, A_F)
            X = X + Delta_X
            self.logger.debug("Updating current X by Delta:_X:\n{}".format(Delta_X))

            F, P = self._get_F_and_P(X)
            A_F = (self.system.A - self.system.B @ F)
            residual = self.get_residual(X, P, F, A_F)
            determinant = linalg.det(P) * determinant_R

    def get_residual(self, X, P, F, A_F):
        res1 = -self.system.B.T @ X + self.system.S.T - self.system.R @ F
        res2 = self.system.Q - self.system.A.T @ X - X @ self.system.A - F.T @ self.system.R @ F - P
        res3 = P @ A_F
        res3 = res3 + res3.T
        self.logger.debug("\nres1:\n{},\nres2: {},\nres3: {}".format(res1, res2, res3))
        return np.linalg.norm(linalg.block_diag(res1, res2, res3))

    def _get_ascent_direction(self, X, A_F):
        A_T = rsolve(self.riccati_operator(X), A_F)
        self.logger.debug("Current Feedback Matrix A_F:\n{}".format(A_F))
        self.logger.debug("Current Feedback Matrix transformed A_T:\n{}".format(A_T))
        A_T_symmetric = A_T + np.asmatrix(A_T).H
        #We're assuming simple eigenvalues here!
        largest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(self.system.n - 1, self.system.n - 1))
        smallest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(0, 0))


        self.logger.debug("Symmetric part of A_T:\n{}".format(A_T_symmetric))
        largest_eigenvector = largest_eigenpair[1]
        largest_eigenvalue = largest_eigenpair[0]
        smallest_eigenvector = smallest_eigenpair[1]
        smallest_eigenvalue = smallest_eigenpair[0]

        if np.abs(smallest_eigenvalue) < np.abs(largest_eigenvalue):
            largest_abs_eigenvalue = largest_eigenvalue
            largest_abs_eigenvector = largest_eigenvector
        else:
            largest_abs_eigenvalue = smallest_eigenvalue
            largest_abs_eigenvector = smallest_eigenvector
        Delta_X = largest_abs_eigenvector @ np.asmatrix(largest_abs_eigenvector).H

        self.logger.debug(
            "largest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(largest_eigenvalue,
                                                                                       largest_eigenvector,
                                                                                       linalg.norm(
                                                                                           largest_eigenvector)))
        self.logger.debug(
            "smallest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(smallest_eigenvalue,
                                                                                        smallest_eigenvector,
                                                                                        linalg.norm(
                                                                                            smallest_eigenvector)))
        # if self.debug:
        #     self.det_direction_plot(X, Delta_X)
        ipdb.set_trace()
        stepsize = self._get_ascent_step_size(X, largest_abs_eigenvector)
        return stepsize * Delta_X

    def _get_ascent_step_size(self, X, eigenvector):
        # perm = np.matrix([[0, 1], [1, 0]])
        H = self._get_H_matrix(X)
        eigenvector = np.asmatrix(eigenvector)  # / linalg.norm(eigenvector)
        self.logger.debug("H(X): {}".format(H))
        ProjectionMatrix = np.asmatrix(linalg.block_diag(eigenvector, eigenvector)).H @ np.bmat(
            [[self.system.A, self.system.B],
             [np.identity(self.system.n), np.zeros((self.system.n, self.system.m))]])

        ProjectedMatrix = ProjectionMatrix @ linalg.solve(H, ProjectionMatrix.H)
        c = ProjectedMatrix[1, 0]
        ab = ProjectedMatrix[0, 0] * ProjectedMatrix[1, 1]
        stepsize = c / (-ab + c ** 2)
        stepsize = c / (-ab + c ** 2)
        # ipdb.set_trace()
        self.logger.debug("Chosen stepsize: {}".format(stepsize))
        return stepsize

    def riccati_operator(self, X):
        RF = - self.system.B.T @ X + self.system.S.T
        Ricc = self.system.Q - self.system.A.T @ X - X @ self.system.A - RF.T @ linalg.solve(self.system.R, RF)
        return Ricc

    def det_direction_plot(self, X, Delta_X):
        self.logger.debug("Creating det direction plot...")
        alpha = np.linspace(-1., 1., 10000)
        det_alpha = [linalg.det(self.riccati_operator(X + val * Delta_X)) for val in alpha]
        # ipdb.set_trace()
        plt.plot(alpha, det_alpha)
        self.logger.debug("Maximum reached at alpha = {}".format(alpha[np.argmax(det_alpha)]))
        # plt.show()
        return alpha[np.argmax(det_alpha)]

    def gradient_sweep(self, X):
        H = self._get_H_matrix(X)
        Hinv = np.asmatrix(linalg.inv(H))
        dimension = self.system.n
        grad = np.zeros((dimension, dimension))
        for i in np.arange(dimension):
            for j in np.arange(i + 1):
                E = np.zeros((dimension, dimension))
                if i == j:
                    E[i, i] = 1
                else:
                    E[i, j] = 1 / np.sqrt(2)
                    E[j, i] = 1 / np.sqrt(2)
                Delta_H = self._get_Delta_H(E)
                grad[i, j] = np.trace(- Hinv.H @ Delta_H)
                self.logger.debug("Gradient: {} Direction: {}, {}".format(grad[i, j], i, j))
        return np.unravel_index(np.argmax(np.abs(grad)), grad.shape)


if __name__ == "__main__":
    logging_config = load_config()
    prepare_logger(logging_config)
    A = np.matrix([[1, -1], [1, 1]])
    B = np.matrix([[1], [1]])
    C = B.T
    D = C @ B
    Q = np.identity(2)
    sys = OptimalControlSystem(A, B, C, D, Q, 0 * B, 2 * D)
    alg = AnalyticCenter(sys, 10 ** (-3))
    alg.steepest_ascent()
