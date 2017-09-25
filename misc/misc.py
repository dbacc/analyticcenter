import numpy as np
from scipy import linalg
import logging
logger = logging.getLogger()



def rsolve(*args, **kwargs):
    """Computes the left inverse.
    
    rsolve(A,B) = solve(A.H, B.H).H = (inv(A.H) @ B.H).H = B @ inv(A)"""
    return np.asmatrix(linalg.solve(np.asmatrix(args[0]).H, np.asmatrix(args[1]).H, kwargs)).H


def schur_complement(X, n, mode='upper'):
    s = X.shape[0]
    if mode.lower() == 'upper':
        complement = X[0:n, 0:n] - X[0:n, n:s] @ linalg.inv(X[n:s, n:s]) @ X[n:s, 0:n]
    else:
        raise NotImplementedError("Schur Complement currently only works for upper part")
    return complement


def check_positivity(M, M_name=""):
    logger.debug("Checking positive definiteness for matrix {}:\n{}".format(M_name, M))
    try:
        linalg.cholesky(M)
        logger.debug('Matrix {} is non-negative'.format(M_name))
        return True
    except linalg.LinAlgError as err:
        lmin = np.min(linalg.eigh(M)[0])
        if lmin >=0 or np.isclose(lmin,0):
            logger.warning('Matrix {} seems to be non-negative, but Cholesky factorization failed due to roundoff errors'.format(M_name))
            return True
        else:
            logger.critical('Matrix {} is not non-negative'.format(M_name))
        return False