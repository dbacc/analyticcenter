import numpy as np
from scipy import linalg
import logging
import control
import misc.misc as misc
import scipy.io



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


class WeightedSystem(LTI):
    """Describes LTI system with weight Matrix"""

    def __init__(self, A, B, C, D, Q, S, R):
        super().__init__(A, B, C, D)
        self.Q = np.asmatrix(Q)
        self.S = np.asmatrix(S)
        self.R = np.asmatrix(R)
        self.Rsqrtm = np.asmatrix(linalg.sqrtm(self.R))
        self.__initH0()


    def save(self):
        np.save('example-n-{}-m-{}'.format(self.n, self.m), [self.A, self.B, self.C, self.D, self.Q, self.S, self.R])

    def save_mat(self):
        scipy.io.savemat('example-n-{}-m-{}'.format(self.n, self.m),
                         {"A": self.A, "B": self.B, "C": self.C, "D": self.D, "Q": self.Q, "S": self.S, "R": self.R})

    def __initH0(self):
        self.H0 = np.bmat([[self.Q, self.S], [self.S.transpose(), self.R]])

