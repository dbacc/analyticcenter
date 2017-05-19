import logging

import numpy as np
from scipy import linalg


def rsolve(*args, **kwargs):
    return linalg.solve(args[0].T, args[1].T, kwargs)


def run():
    A = - np.identity(2)
    B = np.array([[1], [0]])
    C = B.T
    D = C @ B
    sys = OptimalControlSystem(A, B, C, D, -A, 0 * B, D)
    alg = AnalyticCenter(sys, 10 ** (-1))


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
        self.__init_H0()
        self.logger = logging.getLogger(__name__)

    def __init_H0(self):
        self.H0 = np.bmat([[self.system.Q, self.system.S], [self.system.S.transpose(), self.system.R]])

    def build_H_matrix(self, X: np.matrix):
        A = self.system.A
        B = self.system.B
        H = self.H0 + np.bmat([[A.T @ X + X @ A, X @ B],
                               [B.T @ X, np.zeros(self.system.m)]])
        return H

    def _get_initial_X(self):
        self.logger.info('Computing initial X')
        X_plus = linalg.solve_continuous_are(self.system.A, self.system.B, self.system.Q, self.system.R)
        X_minus = linalg.solve_continuous_are(- self.system.A, self.system.B, self.system.Q, self.system.R)
        return 0.5 * (X_minus + X_plus)

    def _get_F_and_P(self, X):
        F = linalg.solve(self.system.R, self.system.S.H - self.system.B.H @ X)
        P = self.system.Q + self.system.A.T @ X + X @ self.system.A - F.T @ self.system.R @ F
        return F, P

    def steepest_ascent(self):
        # noinspection PyPep8Naming
        X = self._get_initial_X()
        F, P = self._get_F_and_P(X)
        steps_count = 0
        while self.get_residual(X, P, F) > self.tol:
            steps_count += 1
            Delta_X = self._get_ascent_direction(X, P, F)
            X = X + Delta_X
            F, P = self._get_F_and_P(X)

    def get_residual(self, X, P, F):
        A_F = (self.system.A - self.system.B @ F)
        res1 = self.system.B.T @ X + self.system.S.T - self.system.R @ F
        res2 = self.system.Q + self.system.A.T @ X + X @ self.system.A - F.T @ self.system.R @ F - P
        res3 = P @ A_F
        res3 = res3 + res3.T
        return np.norm([res1, res2, res3])

    def _get_ascent_direction(self, X, A_F):
        T = linalg.sqrtm(self.riccati_operator(X))
        A_T = T @ rsolve(A_F, T)
        largest_eigenvector = linalg.eigh(A_T + A_T.T, eigvals=(self.system.n - 1,))[1][:, 0]
        Delta_T = largest_eigenvector @ largest_eigenvector.T
        Delta_X = T @ Delta_T @ T
        stepzsize = self._get_ascent_step_size(X, largest_eigenvector)
        return stepzsize * Delta_X

    def _get_ascent_step_size(self, X, eigenvector):
        # perm = np.matrix([[0, 1], [1, 0]])
        H = self.build_H_matrix(X)
        ProjectionMatrix = linalg.block_diag(eigenvector, eigenvector) @ np.bmat(
            [[self.system.A, self.system.B], np.identity(2), np.zeros(2, self.n)])
        ProjectedMatrix = ProjectionMatrix.H @ linalg.solve(H, ProjectionMatrix)
        return 0.5 * np.trace(ProjectedMatrix)

    def riccati_operator(self, X):
        RF = self.system.B.T @ X + self.system.S.T
        Ricc = self.system.Q + self.system.A.T @ X + X @ self.system.A - RF.T @ linalg.solve(self.system.R, RF)
        return Ricc


if __name__ == "__main__":
    run()


def gradient():
    pass
