import logging

import numpy as np
from scipy import linalg

from analyticcenter.direction import DirectionAlgorithm
import misc.misc as misc


def get_analytic_center_object(system, tol, discrete_time=False):
    if discrete_time:
        return AnalyticCenterDiscreteTime(system, tol)
    else:
        return AnalyticCenterContinuousTime(system, tol)


class AnalyticCenter(object):
    """ToDo"""
    debug = True
    logger = logging.getLogger(__name__)
    rel_tol = 1.e-16
    def __init__(self, system, tol, discrete_time):
        self.system = system
        self.X0 = None
        self.H0 = None
        self.H = None
        self.tol = tol
        self.maxiter = 100
        self.__init_H0()
        self.debug = True
        DirectionAlgorithm(self, self.discrete_time)

    def __init_H0(self):
        self.H0 = np.bmat([[self.system.Q, self.system.S], [self.system.S.H, self.system.R]])

    def _get_Delta_H(self, Delta_X):
        return self._get_H_matrix(Delta_X) - self.H0

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


class AnalyticCenterContinuousTime(AnalyticCenter):
    # TODO: Improve performance by saving intermediate results where appropriate
    discrete_time = False

    def __init__(self, system, tol):
        super().__init__(system, tol, False)
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

    def get_residual(self, X, P, F, A_F, Delta = None):

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
            Ricc -= - RF.H @ linalg.solve(self.system.R, RF)
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

class AnalyticCenterDiscreteTime(AnalyticCenter):
    # TODO: Improve performance by saving intermediate results where appropriate
    discrete_time = True

    def __init__(self, system, tol):
        super().__init__(system, tol, True)

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


