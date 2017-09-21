import numpy as np
from scipy import linalg
import logging
import control
import misc.misc as misc
import scipy.io
from misc.control import place


class LTI(object):
    """Describes an LTI system"""

    def __init__(self, A, B, C, D):
        self.A = np.asmatrix(A)
        self.B = np.asmatrix(B)
        self.C = np.asmatrix(C)
        self.D = np.asmatrix(D)
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.p = C.shape[0]
        self.logger = logging.getLogger(__name__)


class OptimalControlSystem(LTI):
    """Describes LTI system with weight Matrix"""

    def __init__(self, A, B, C, D, Q, S, R):
        super().__init__(A, B, C, D)
        self.Q = np.asmatrix(Q)
        self.S = np.asmatrix(S)
        self.R = np.asmatrix(R)
        self.__initH0()
        misc.check_positivity(self.H0, "H_0")

    def save(self):
        np.save('example-n-{}-m-{}'.format(self.n, self.m), [self.A, self.B, self.C, self.D, self.Q, self.S, self.R])

    def save_mat(self):
        scipy.io.savemat('example-n-{}-m-{}'.format(self.n, self.m),
                         {"A": self.A, "B": self.B, "C": self.C, "D": self.D, "Q": self.Q, "S": self.S, "R": self.R})

    def __initH0(self):
        self.H0 = np.bmat([[self.Q, self.S], [self.S.transpose(), self.R]])

    def check_stability(self):
        eigs = np.linalg.eig(self.system.A)[0]
        real = np.real(eigs)
        if np.max(real) < 0:
            self.logger.info("System is stable")
            return True
        else:
            self.logger.critical("System is not stable. Aborting.")
            raise BaseException("System is not stable.")
            return False

    def check_controllability(self):
        poles = np.random.rand(self.n)
        F, nup = place(self.A, self.B, poles)
        if nup >0:
            self.logger.critical("System is not controllable. Aborting.")
            raise BaseException("System is not stable.")
            return False
        else:
            self.logger.info("System is controllable.")
            return True


    def check_passivity(self):
        ricc = control.care(self.A, self.B, self.Q, self.R, self.S, np.identity(self.n))
        if misc.check_positivity(ricc[0], 'X'):
            self.logger.info("System is passive, if also stable")
        else:
            self.logger.critical("System cannot be stabilized (in particular unstable)")
            raise BaseException("System cannot be stabilized (in particular unstable)")
