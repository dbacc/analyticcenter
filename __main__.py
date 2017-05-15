import numpy as np


def run():
    pass


if __name__ == "__main__":
    run()


def gradient():
    pass


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


class OptimalControlSystem(LTI):
    """Describes LTI system with weight Matrix"""

    def __init__(self, A, B, C, D, Q, S, R):
        super().__init__(A, B, C, D)
        self.Q = Q
        self.S = S
        self.R = R


class AnalyticCenter(object):
    """ToDo"""

    def __init__(self, system, tol):
        self.system = system
        self.H = None
        self.tol = tol
        __init_H0(self)

    def __init_H0(self):
        self.H0 = np.bmat([[self.system.Q, self.system.S], [self.system.S.transpose(), self.system.R]])

    def build_H_matrix(self, X : np.matrix):
        A = self.system.A
        B = self.system.B
        self.H = self.H0 + np.bmat([[A.T * X + X * A, X * B],
                                    [B.T * X, np.zeros(self.system.m)]])
    def steepest_ascent(self, X0, P0, F0):
        X, P, F = X0, P0, F0

        while self.get_residual(X, P, F) > self.tol:
            self.get_ascent_direction()
    def get_residual(self, X, P, F):
        res1 = self.system.B.T*X + self.system.S.T - self.system.R * F
        res2 = self.system.Q + self.system.A.T * X + X * self.system.A - F.T * self.system.R * F
        res3 = P * (self.system.A - self.system.B * F)
        res3 = res3 + res3.T
        return np.norm([res1, res2, res3])
    def _get_ascent_direction(self):
        T =
