import numpy as np
import scipy.linalg as linalg
import logging
from ..misc.misc import rsolve
from slycot import ab13fd

class AnalyticCenter(object):
    """Class for storing the final solution X at the analytic center.
    Contains convenience functions for computing the characteristic values in [].
    """
    def __init__(self, X, A_F, HX, algorithm=None, discrete_time=False, delta_cum=None):
        self.algorithm = algorithm
        self.X = X
        self.A_F = A_F
        self.HX = HX
        self.discrete_time = discrete_time
        self.system = algorithm.system
        self.logger = logging.getLogger(__name__)
        self.delta_cum = delta_cum

    def lambda_min_alpha(self):
        """ """
        eigs = linalg.eigh(self.HX)[0]
        eigmin = np.min(eigs)
        self.logger.info("Minimal eigenvalue of H(X): {}".format(eigmin))
        return eigmin

    def lambda_min_beta(self):
        """ """
        X2 = self.X @ self.X
        X2id = linalg.block_diag(self.X, np.identity(self.system.m))
        M = rsolve(X2id, linalg.solve(X2id, self.HX))
        eigs = linalg.eigh(M)[0]
        eigmin = np.min(eigs)
        self.logger.info("Minimal eigenvalue of X-1 * H(X) * X^-1: {}".format(eigmin))
        return eigmin

    def lambda_min_xi(self):
        """ """
        Xhat = linalg.block_diag(self.X, np.identity(self.system.m))
        Xhathalf = linalg.sqrtm(Xhat)
        M = rsolve(Xhathalf, linalg.solve(Xhathalf, self.HX))
        eigs = linalg.eigh(M)[0]
        eigmin = np.min(eigs)
        self.logger.info("Minimal eigenvalues of X^-.5 * H(X) * X^-.5: {}".format(eigmin))
        return eigmin

    def stability_estimates(self):
        """ """
        T = linalg.sqrtm(self.X)
        A_T = np.asmatrix(linalg.solve(T, self.system.A) @ T)
        R =-( A_T + A_T.H)
        eigminR = np.min(linalg.eigh(R)[0])
        self.logger.info("Minimal eigenvalues of R of the pH realization: {}".format(eigminR))
        dist, omega = ab13fd(self.system.n, A_T, tol=1.e-5)
        self.logger.info("Distance to stability is: {}".format(dist))


    def compute_characteristic_values(self):
        """ """
        self.lambda_min_alpha()
        self.lambda_min_beta()
        self.lambda_min_xi()
        self.stability_estimates()
