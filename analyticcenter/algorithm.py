import logging

import numpy as np
from scipy import linalg

from analyticcenter.direction import DirectionAlgorithm
import misc.misc as misc
import ipdb
from misc.control import place
import control
from .exceptions import AnalyticCenterNotPassive, AnalyticCenterUncontrollable, AnalyticCenterUnstable, AnalyticCenterRiccatiSolutionFailed

logger = logging.getLogger(__name__)


def get_analytic_center_object(*args, **kwargs):
    discrete_time = kwargs.get('discrete_time')
    if discrete_time is None:
        discrete_time = False
        logger.warning("No system type given. Defaulting to continuous time.")
    if discrete_time:
        return AlgorithmDiscreteTime(*args, **kwargs)
    else:
        return AlgorithmContinuousTime(*args, **kwargs)


class Algorithm(object):
    """ToDo"""
    debug = True
    logger = logging.getLogger(__name__)

    def __init__(self, system, abs_tol=1.e-5, rel_tol=1.e-8, discrete_time=False):
        self.system = system
        self.X0 = None
        self.H0 = None
        self.H = None
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.maxiter = 100
        self.__init_H0()
        self.debug = True
        self.check_stability()
        self.check_controllability()
        self.check_passivity()
        DirectionAlgorithm(self, self.discrete_time)

    def __init_H0(self):
        self.H0 = np.bmat([[self.system.Q, self.system.S], [self.system.S.H, self.system.R]])

    def _get_Delta_H(self, Delta_X):
        return self._get_H_matrix(Delta_X) - self.H0

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

    def _get_H_matrix(self, X: np.matrix):
        raise NotImplementedError

    def _get_F_and_P(self, X):
        raise NotImplementedError

    def get_residual(self, X, P, F, A_F):
        raise NotImplementedError

    def riccati_operator(self, X, F=None):
        raise NotImplementedError

    def _get_R(self, X):
        raise NotImplementedError

    def _get_determinant_R(self, X):
        raise NotImplementedError

    def _eig_stable(self):
        raise NotImplementedError

    def check_stability(self):
        eigs = np.linalg.eig(self.system.A)[0]
        if self._eig_stable(eigs):
            self.logger.info("System is stable")
            return True
        else:
            self.logger.critical("System is not stable. Aborting.")
            raise AnalyticCenterUnstable("System is not stable.")
            return False

    def check_controllability(self):
        poles = np.random.rand(self.system.n)
        F, nup = place(self.system.A, self.system.B, poles)
        if nup > 0:
            self.logger.critical("System is not controllable. Aborting.")
            raise AnalyticCenterUncontrollable("System is not controllable.")
            return False
        else:
            self.logger.info("System is controllable.")
            return True

    def check_passivity(self):
        try:
            ricc = self.riccati_solver(self.system.A, self.system.B, self.system.Q, self.system.R, self.system.S,
                                   np.identity(self.system.n))


            X = - ricc[0]

            if misc.check_positivity(self._get_H_matrix(X), 'X'):
                self.logger.info("System is passive, if also stable")
            else:
                self.logger.critical("System is not passive")
                raise AnalyticCenterNotPassive("System is not passive")
        except ValueError as e:
            self.logger.critical("Riccati solver for passivity check did not succeed with message:\n{}".format(e.args[0]))
            raise AnalyticCenterRiccatiSolutionFailed("Riccati solver for passivity check did not succeed")


class AlgorithmContinuousTime(Algorithm):
    # TODO: Improve performance by saving intermediate results where appropriate
    discrete_time = False
    riccati_solver = staticmethod(control.care)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._determinant_R = None

    def _get_H_matrix(self, X: np.matrix):
        A = self.system.A
        B = self.system.B
        H = self.H0 - np.bmat([[A.H @ X + X @ A, X @ B],
                               [B.H @ X, np.zeros((self.system.m, self.system.m))]])
        if self.debug:
            misc.check_positivity(H, "H(X)")
        return H

    def _get_F_and_P(self, X):
        F = np.asmatrix(linalg.solve(self.system.R, self.system.S.H - self.system.B.H @ X))
        P = self.riccati_operator(X, F)
        return F, P

    def get_residual(self, X, P, F, A_F, Delta=None):

        res = P @ A_F
        res = res + res.H
        self.logger.debug("res: {}".format(res))
        if Delta is None:
            return np.linalg.norm(res)
        else:
            return np.real(np.trace(res @ Delta))

    def riccati_operator(self, X, F=None):
        RF = - self.system.B.H @ X + self.system.S.H
        Ricc = self.system.Q - self.system.A.H @ X - X @ self.system.A
        if F is None:
            Ricc -= RF.H @ linalg.solve(self.system.R, RF)
        else:
            Ricc -= F.H @ self.system.R @ F
        return Ricc

    def _get_R(self, X):
        return self.system.R

    def _get_determinant_R(self, X):
        if self._determinant_R is None:
            self._determinant_R = linalg.det(self.system.R)
        return self._determinant_R

    def _get_Hamiltonian(self):
        A = self.system.A
        B = self.system.B
        Q = self.system.Q
        R = self.system.R
        S = self.system.S
        RinvSH = linalg.solve(R, S.H)
        H1 = A - B @ RinvSH
        Ham = np.bmat([[H1, - B @ linalg.solve(R, B.H)],
                       [-Q + S @ RinvSH, -H1.H]])
        self.logger.debug("Eigenvalues of the Hamiltonian:\n{}".format(linalg.eig(Ham)[0]))

    def _eig_stable(self, eigs):
        return np.max(np.real(eigs)) < 0


class AlgorithmDiscreteTime(Algorithm):
    # TODO: Improve performance by saving intermediate results where appropriate
    discrete_time = True
    riccati_solver = staticmethod(control.dare)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_H_matrix(self, X: np.matrix):
        A = self.system.A
        B = self.system.B
        H = self.H0 - np.bmat([[A.H @ X @ A - X, A.H @ X @ B],
                               [B.H @ X @ A, B.H @ X @ B]])
        if self.debug:
            misc.check_positivity(H, "H(X)")
        return H

    def _get_F_and_P(self, X):
        F = np.asmatrix(linalg.solve(self.system.R - self.system.B.H @ X @ self.system.B,
                                     self.system.S.H - self.system.B.H @ X @ self.system.A))
        P = self.riccati_operator(X, F)
        return F, P

    def get_residual(self, X, P, F, A_F, Delta=None):
        Pinv = linalg.inv(P)
        res = A_F @ Pinv @ A_F.H - Pinv + self.system.B @ linalg.solve(
            self.system.R - self.system.B.H @ X @ self.system.B,
            self.system.B.H)
        self.logger.debug("res: {}".format(res))
        if Delta is None:
            return np.linalg.norm(res)
        else:
            return np.real(np.trace(res @ Delta))

    def riccati_operator(self, X, F=None):
        RF = - self.system.B.H @ X @ self.system.A + self.system.S.H
        Ricc = self.system.Q - self.system.A.H @ X @ self.system.A + X
        R = self._get_R(X)
        if F is None:
            Ricc -= RF.H @ linalg.solve(R, RF)
        else:
            Ricc -= F.H @ R @ F
        return Ricc

    def _get_R(self, X):
        return self.system.R - self.system.B.H @ X @ self.system.B

    def _get_determinant_R(self, X):
        return linalg.det(self.system.R - self.system.B.H @ X @ self.system.B)

    def _eig_stable(self, eigs):
        return np.max(np.abs(eigs)) < 1
