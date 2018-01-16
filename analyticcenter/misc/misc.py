##
## Copyright (c) 2017
## 
## @author: Daniel Bankmann
## @company: Technische Universität Berlin
## 
## This file is part of the python package analyticcenter
## (see https://gitlab.tu-berlin.de/PassivityRadius/analyticcenter/)
## 
## License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
##
import numpy as np
from scipy import linalg
import logging
import ipdb

logger = logging.getLogger(__name__)


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
        if lmin >= 0 or np.isclose(lmin, 0):
            logger.debug(
                'Matrix {} seems to be non-negative, but Cholesky factorization failed due to roundoff errors'.format(
                    M_name))
            return True
        else:
            logger.critical('Matrix {} is not non-negative'.format(M_name))
        return False


def symmetric_product_pos_def(B, P, invertP=False):
    """Computes the product B.H @ P.I @ B in a symmetry-preserving way
    input:
        P: pos. def. 2d-array
        B: 2d array
    output:
        B.H @ P.I @ B with B.H @ P.I @ B - (B.H @ P.I @ B).H = 0
    """
    T, Z = linalg.schur(P)
    Z = np.asmatrix(Z)
    if invertP:
        D = np.diag(1 / np.sqrt(np.diag(T)))  # force diagonal matrix
    else:
        D = np.diag(np.sqrt(np.diag(T)))
    product = D @ Z.H @ B
    product = product.H @ product
    return product
