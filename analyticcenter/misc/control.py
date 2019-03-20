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
""" mateqn.py

Matrix equation solvers (Lyapunov, Riccati)

Implementation of the functions lyap, dlyap, care and dare
for solution of Lyapunov and Riccati equations. """

# Python 3 compatability (needs to go here)
from __future__ import print_function


"""Copyright (c) 2011, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the project author nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Author: Bjorn Olofsson
"""
import numpy as np
from numpy.linalg import inv
from scipy import shape, size, asarray, asmatrix, copy, zeros, eye, dot
from scipy.linalg import eigvals, solve_discrete_are
from control.exception import ControlSlycot, ControlArgument, ControlDimension


#### Riccati equation solvers care and dare

def care(A, B, Q, R=None, S=None, E=None):
    """ (X,L,G) = care(A,B,Q,R=None) solves the continuous-time algebraic Riccati
    equation

        :math:`A^T X + X A - X B R^{-1} B^T X + Q = 0`

    where A and Q are square matrices of the same dimension. Further,
    Q and R are a symmetric matrices. If R is None, it is set to the
    identity matrix. The function returns the solution X, the gain
    matrix G = B^T X and the closed loop eigenvalues L, i.e., the
    eigenvalues of A - B G.

    (X,L,G) = care(A,B,Q,R,S,E) solves the generalized continuous-time
    algebraic Riccati equation

        :math:`A^T X E + E^T X A - (E^T X B + S) R^{-1} (B^T X E + S^T) + Q = 0`

    where A, Q and E are square matrices of the same
    dimension. Further, Q and R are symmetric matrices. If R is None,
    it is set to the identity matrix. The function returns the
    solution X, the gain matrix G = R^-1 (B^T X E + S^T) and the
    closed loop eigenvalues L, i.e., the eigenvalues of A - B G , E."""

    # Make sure we can import required slycot routine
    try:
        from slycot import sb02md
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02md'")

    try:
        from slycot import sb02mt
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02mt'")

    # Make sure we can find the required slycot routine
    try:
        from slycot import sg02ad
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sg02ad'")

    # Reshape 1-d arrays
    if len(shape(A)) == 1:
        A = A.reshape(1, A.size)

    if len(shape(B)) == 1:
        B = B.reshape(1, B.size)

    if len(shape(Q)) == 1:
        Q = Q.reshape(1, Q.size)

    if R is not None and len(shape(R)) == 1:
        R = R.reshape(1, R.size)

    if S is not None and len(shape(S)) == 1:
        S = S.reshape(1, S.size)

    if E is not None and len(shape(E)) == 1:
        E = E.reshape(1, E.size)

    # Determine main dimensions
    if size(A) == 1:
        n = 1
    else:
        n = size(A, 0)

    if size(B) == 1:
        m = 1
    else:
        m = size(B, 1)
    if R is None:
        R = eye(m, m)

    # Solve the standard algebraic Riccati equation
    if S is None and E is None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
                (size(Q) > 1 and shape(Q)[0] != n) or \
                                size(Q) == 1 and n > 1:
            raise ControlArgument("Q must be a quadratic matrix of the same \
                dimension as A.")

        if (size(B) > 1 and shape(B)[0] != n) or \
                                size(B) == 1 and n > 1:
            raise ControlArgument("Incompatible dimensions of B matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        if not (asarray(R) == asarray(R).T).all():
            raise ControlArgument("R must be a symmetric matrix.")

        # Create back-up of arrays needed for later computations
        R_ba = copy(R)
        B_ba = copy(B)

        # Solve the standard algebraic Riccati equation by calling Slycot
        # functions sb02mt and sb02md
        try:
            A_b, B_b, Q_b, R_b, L_b, ipiv, oufact, G = sb02mt(n, m, B, R)
        except ValueError as ve:
            if ve.info < 0:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == m + 1:
                e = ValueError("The matrix R is numerically singular.")
                e.info = ve.info
            else:
                e = ValueError("The %i-th element of d in the UdU (LdL) \
                    factorization is zero." % ve.info)
                e.info = ve.info
            raise e

        try:
            X, rcond, w, S_o, U, A_inv = sb02md(n, A, G, Q, 'C')
        except ValueError as ve:
            if ve.info < 0 or ve.info > 5:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The matrix A is (numerically) singular in \
                    continuous-time case.")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The Hamiltonian or symplectic matrix H cannot \
                    be reduced to real Schur form.")
                e.info = ve.info
            elif ve.info == 3:
                e = ValueError("The real Schur form of the Hamiltonian or \
                    symplectic matrix H cannot be appropriately ordered.")
                e.info = ve.info
            elif ve.info == 4:
                e = ValueError("The Hamiltonian or symplectic matrix H has \
                    less than n stable eigenvalues.")
                e.info = ve.info
            elif ve.info == 5:
                e = ValueError("The N-th order system of linear algebraic \
                         equations is singular to working precision.")
                e.info = ve.info
            raise e

        # Calculate the gain matrix G
        if size(R_b) == 1:
            G = dot(dot(1 / (R_ba), asarray(B_ba).T), X)
        else:
            G = dot(dot(inv(R_ba), asarray(B_ba).T), X)

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return (X, w[:n], G)

    # Solve the generalized algebraic Riccati equation
    elif S is not None and E is not None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
                (size(Q) > 1 and shape(Q)[0] != n) or \
                                size(Q) == 1 and n > 1:
            raise ControlArgument("Q must be a quadratic matrix of the same \
                dimension as A.")

        if (size(B) > 1 and shape(B)[0] != n) or \
                                size(B) == 1 and n > 1:
            raise ControlArgument("Incompatible dimensions of B matrix.")

        if (size(E) > 1 and shape(E)[0] != shape(E)[1]) or \
                (size(E) > 1 and shape(E)[0] != n) or \
                                size(E) == 1 and n > 1:
            raise ControlArgument("E must be a quadratic matrix of the same \
                dimension as A.")

        if (size(R) > 1 and shape(R)[0] != shape(R)[1]) or \
                (size(R) > 1 and shape(R)[0] != m) or \
                                size(R) == 1 and m > 1:
            raise ControlArgument("R must be a quadratic matrix of the same \
                dimension as the number of columns in the B matrix.")

        if (size(S) > 1 and shape(S)[0] != n) or \
                (size(S) > 1 and shape(S)[1] != m) or \
                                size(S) == 1 and n > 1 or \
                                size(S) == 1 and m > 1:
            raise ControlArgument("Incompatible dimensions of S matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        if not (asarray(R) == asarray(R).T).all():
            raise ControlArgument("R must be a symmetric matrix.")

        # Create back-up of arrays needed for later computations
        R_b = copy(R)
        B_b = copy(B)
        E_b = copy(E)
        S_b = copy(S)

        # Solve the generalized algebraic Riccati equation by calling the
        # Slycot function sg02ad
        try:
            rcondu, X, alfar, alfai, beta, S_o, T, U, iwarn = \
                sg02ad('C', 'B', 'N', 'U', 'N', 'N', 'U', 'R', n, m, 0, A, E, B, Q, R, S)
        except ValueError as ve:
            if ve.info < 0 or ve.info > 7:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The computed extended matrix pencil is \
                            singular, possibly due to rounding errors.")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The QZ algorithm failed.")
                e.info = ve.info
            elif ve.info == 3:
                e = ValueError("Reordering of the generalized eigenvalues \
                    failed.")
                e.info = ve.info
            elif ve.info == 4:
                e = ValueError("After reordering, roundoff changed values of \
                            some complex eigenvalues so that leading \
                            eigenvalues in the generalized Schur form no \
                            longer satisfy the stability condition; this \
                            could also be caused due to scaling.")
                e.info = ve.info
            elif ve.info == 5:
                e = ValueError("The computed dimension of the solution does \
                            not equal N.")
                e.info = ve.info
            elif ve.info == 6:
                e = ValueError("The spectrum is too close to the boundary of \
                            the stability domain.")
                e.info = ve.info
            elif ve.info == 7:
                e = ValueError("A singular matrix was encountered during the \
                            computation of the solution matrix X.")
                e.info = ve.info
            raise e

        # Calculate the closed-loop eigenvalues L
        L = zeros((n, 1))
        L.dtype = 'complex64'
        for i in range(n):
            L[i] = (alfar[i] + alfai[i] * 1j) / beta[i]

        # Calculate the gain matrix G
        if size(R_b) == 1:
            G = dot(1 / (R_b), dot(asarray(B_b).T, dot(X, E_b)) + asarray(S_b).T)
        else:
            G = dot(inv(R_b), dot(asarray(B_b).T, dot(X, E_b)) + asarray(S_b).T)

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return (X, L, G)

    # Invalid set of input parameters
    else:
        raise ControlArgument("Invalid set of input parameters.")


def dare(A, B, Q, R, S=None, E=None, stabilizing=True):
    """ (X,L,G) = dare(A,B,Q,R) solves the discrete-time algebraic Riccati
    equation

        :math:`A^T X A - X - A^T X B (B^T X B + R)^{-1} B^T X A + Q = 0`

    where A and Q are square matrices of the same dimension. Further, Q
    is a symmetric matrix. The function returns the solution X, the gain
    matrix G = (B^T X B + R)^-1 B^T X A and the closed loop eigenvalues L,
    i.e., the eigenvalues of A - B G.

    (X,L,G) = dare(A,B,Q,R,S,E) solves the generalized discrete-time algebraic
    Riccati equation

        :math:`A^T X A - E^T X E - (A^T X B + S) (B^T X B + R)^{-1} (B^T X A + S^T) + Q = 0`

    where A, Q and E are square matrices of the same dimension. Further, Q and
    R are symmetric matrices. The function returns the solution X, the gain
    matrix :math:`G = (B^T X B + R)^{-1} (B^T X A + S^T)` and the closed loop
    eigenvalues L, i.e., the eigenvalues of A - B G , E.
    """
    if S is not None or E is not None or not stabilizing:
        return dare_old(A, B, Q, R, S, E, stabilizing)
    else:
        Rmat = asmatrix(R)
        Qmat = asmatrix(Q)
        X = solve_discrete_are(A, B, Qmat, Rmat)
        G = inv(B.T.dot(X).dot(B) + Rmat) * B.T.dot(X).dot(A)
        L = eigvals(A - B.dot(G))
        return X, L, G


def dare_old(A, B, Q, R, S=None, E=None, stabilizing=True):
    # Make sure we can import required slycot routine
    try:
        from slycot import sb02md
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02md'")

    try:
        from slycot import sb02mt
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02mt'")

    # Make sure we can find the required slycot routine
    try:
        from slycot import sg02ad
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sg02ad'")

    # Reshape 1-d arrays
    if len(shape(A)) == 1:
        A = A.reshape(1, A.size)

    if len(shape(B)) == 1:
        B = B.reshape(1, B.size)

    if len(shape(Q)) == 1:
        Q = Q.reshape(1, Q.size)

    if R is not None and len(shape(R)) == 1:
        R = R.reshape(1, R.size)

    if S is not None and len(shape(S)) == 1:
        S = S.reshape(1, S.size)

    if E is not None and len(shape(E)) == 1:
        E = E.reshape(1, E.size)

    # Determine main dimensions
    if size(A) == 1:
        n = 1
    else:
        n = size(A, 0)

    if size(B) == 1:
        m = 1
    else:
        m = size(B, 1)

    # Solve the standard algebraic Riccati equation
    if S is None and E is None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
                (size(Q) > 1 and shape(Q)[0] != n) or \
                                size(Q) == 1 and n > 1:
            raise ControlArgument("Q must be a quadratic matrix of the same \
                dimension as A.")

        if (size(B) > 1 and shape(B)[0] != n) or \
                                size(B) == 1 and n > 1:
            raise ControlArgument("Incompatible dimensions of B matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        if not (asarray(R) == asarray(R).T).all():
            raise ControlArgument("R must be a symmetric matrix.")

        # Create back-up of arrays needed for later computations
        A_ba = copy(A)
        R_ba = copy(R)
        B_ba = copy(B)

        # Solve the standard algebraic Riccati equation by calling Slycot
        # functions sb02mt and sb02md
        try:
            A_b, B_b, Q_b, R_b, L_b, ipiv, oufact, G = sb02mt(n, m, B, R)
        except ValueError as ve:
            if ve.info < 0:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == m + 1:
                e = ValueError("The matrix R is numerically singular.")
                e.info = ve.info
            else:
                e = ValueError("The %i-th element of d in the UdU (LdL) \
                     factorization is zero." % ve.info)
                e.info = ve.info
            raise e

        try:
            if stabilizing:
                sort = 'S'
            else:
                sort = 'U'

            X, rcond, w, S, U, A_inv = sb02md(n, A, G, Q, 'D', sort=sort)
        except ValueError as ve:
            if ve.info < 0 or ve.info > 5:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The matrix A is (numerically) singular in \
                    discrete-time case.")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The Hamiltonian or symplectic matrix H cannot \
                    be reduced to real Schur form.")
                e.info = ve.info
            elif ve.info == 3:
                e = ValueError("The real Schur form of the Hamiltonian or \
                     symplectic matrix H cannot be appropriately ordered.")
                e.info = ve.info
            elif ve.info == 4:
                e = ValueError("The Hamiltonian or symplectic matrix H has \
                     less than n stable eigenvalues.")
                e.info = ve.info
            elif ve.info == 5:
                e = ValueError("The N-th order system of linear algebraic \
                     equations is singular to working precision.")
                e.info = ve.info
            raise e

        # Calculate the gain matrix G
        if size(R_b) == 1:
            G = dot(1 / (dot(asarray(B_ba).T, dot(X, B_ba)) + R_ba), \
                    dot(asarray(B_ba).T, dot(X, A_ba)))
        else:
            G = dot(inv(dot(asarray(B_ba).T, dot(X, B_ba)) + R_ba), \
                    dot(asarray(B_ba).T, dot(X, A_ba)))

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return (X, w[:n], G)

    # Solve the generalized algebraic Riccati equation
    elif S is not None and E is not None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
                (size(Q) > 1 and shape(Q)[0] != n) or \
                                size(Q) == 1 and n > 1:
            raise ControlArgument("Q must be a quadratic matrix of the same \
                dimension as A.")

        if (size(B) > 1 and shape(B)[0] != n) or \
                                size(B) == 1 and n > 1:
            raise ControlArgument("Incompatible dimensions of B matrix.")

        if (size(E) > 1 and shape(E)[0] != shape(E)[1]) or \
                (size(E) > 1 and shape(E)[0] != n) or \
                                size(E) == 1 and n > 1:
            raise ControlArgument("E must be a quadratic matrix of the same \
                dimension as A.")

        if (size(R) > 1 and shape(R)[0] != shape(R)[1]) or \
                (size(R) > 1 and shape(R)[0] != m) or \
                                size(R) == 1 and m > 1:
            raise ControlArgument("R must be a quadratic matrix of the same \
                dimension as the number of columns in the B matrix.")

        if (size(S) > 1 and shape(S)[0] != n) or \
                (size(S) > 1 and shape(S)[1] != m) or \
                                size(S) == 1 and n > 1 or \
                                size(S) == 1 and m > 1:
            raise ControlArgument("Incompatible dimensions of S matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        if not (asarray(R) == asarray(R).T).all():
            raise ControlArgument("R must be a symmetric matrix.")

        # Create back-up of arrays needed for later computations
        A_b = copy(A)
        R_b = copy(R)
        B_b = copy(B)
        E_b = copy(E)
        S_b = copy(S)

        # Solve the generalized algebraic Riccati equation by calling the
        # Slycot function sg02ad
        try:
            if stabilizing:
                sort = 'S'
            else:
                sort = 'U'
            rcondu, X, alfar, alfai, beta, S_o, T, U, iwarn = \
                sg02ad('D', 'B', 'N', 'U', 'N', 'N', sort, 'R', n, m, 0, A, E, B, Q, R, S)
        except ValueError as ve:
            if ve.info < 0 or ve.info > 7:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The computed extended matrix pencil is \
                            singular, possibly due to rounding errors.")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The QZ algorithm failed.")
                e.info = ve.info
            elif ve.info == 3:
                e = ValueError("Reordering of the generalized eigenvalues \
                     failed.")
                e.info = ve.info
            elif ve.info == 4:
                e = ValueError("After reordering, roundoff changed values of \
                            some complex eigenvalues so that leading \
                            eigenvalues in the generalized Schur form no \
                            longer satisfy the stability condition; this \
                            could also be caused due to scaling.")
                e.info = ve.info
            elif ve.info == 5:
                e = ValueError("The computed dimension of the solution does \
                            not equal N.")
                e.info = ve.info
            elif ve.info == 6:
                e = ValueError("The spectrum is too close to the boundary of \
                            the stability domain.")
                e.info = ve.info
            elif ve.info == 7:
                e = ValueError("A singular matrix was encountered during the \
                            computation of the solution matrix X.")
                e.info = ve.info
            raise e

        L = zeros((n, 1))
        L.dtype = 'complex64'
        for i in range(n):
            L[i] = (alfar[i] + alfai[i] * 1j) / beta[i]

        # Calculate the gain matrix G
        if size(R_b) == 1:
            G = dot(1 / (dot(asarray(B_b).T, dot(X, B_b)) + R_b), \
                    dot(asarray(B_b).T, dot(X, A_b)) + asarray(S_b).T)
        else:
            G = dot(inv(dot(asarray(B_b).T, dot(X, B_b)) + R_b), \
                    dot(asarray(B_b).T, dot(X, A_b)) + asarray(S_b).T)

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return (X, L, G)

    # Invalid set of input parameters
    else:
        raise ControlArgument("Invalid set of input parameters.")


# def place(A, B, p):
#     """Place closed loop eigenvalues
#
#     Parameters
#     ----------
#     A : 2-d array
#         Dynamics matrix
#     B : 2-d array
#         Input matrix
#     p : 1-d list
#         Desired eigenvalue locations
#
#     Returns
#     -------
#     K : 2-d array
#         Gains such that A - B K has given eigenvalues
#
#     Examples
#     --------
#     >>> A = [[-1, -1], [0, 1]]
#     >>> B = [[0], [1]]
#     >>> K = place(A, B, [-2, -5])
#     """
#
#     # Make sure that SLICOT is installed
#     try:
#         from slycot import sb01bd
#     except ImportError:
#         raise ControlSlycot("can't find slycot module 'sb01bd'")
#
#     # Convert the system inputs to NumPy arrays
#     A_mat = np.array(A);
#     B_mat = np.array(B);
#     if (A_mat.shape[0] != A_mat.shape[1] or
#                 A_mat.shape[0] != B_mat.shape[0]):
#         raise ControlDimension("matrix dimensions are incorrect")
#
#     # Compute the system eigenvalues and convert poles to numpy array
#     system_eigs = np.linalg.eig(A_mat)[0]
#     placed_eigs = np.array(p);
#
#     # SB01BD sets eigenvalues with real part less than alpha
#     # We want to place all poles of the system => set alpha to minimum
#     alpha = min(system_eigs.real);
#
#     # Call SLICOT routine to place the eigenvalues
#     A_z, w, nfp, nap, nup, F, Z, warn = \
#         sb01bd(B_mat.shape[0], B_mat.shape[1], len(placed_eigs), alpha,
#                A_mat, B_mat, placed_eigs, 'C');
#     # Return the gain matrix, with MATLAB gain convention
#     return -F, nup, warn

def place(A, B, p): #Taken from python-control package
    """Place closed loop eigenvalues
    K = place_varga(A, B, p)

    Parameters
    ----------
    A : 2-d array
        Dynamics matrix
    B : 2-d array
        Input matrix
    p : 1-d list
        Desired eigenvalue locations
    Returns
    -------
    K : 2-d array
        Gain such that A - B K has eigenvalues given in p.


    Algorithm
    ---------
        This function is a wrapper for the slycot function sb01bd, which
        implements the pole placement algorithm of Varga [1]. In contrast to
        the algorithm used by place(), the Varga algorithm can place
        multiple poles at the same location. The placement, however, may not
        be as robust.

        [1] Varga A. "A Schur method for pole assignment."
            IEEE Trans. Automatic Control, Vol. AC-26, pp. 517-519, 1981.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place(A, B, [-2, -5])

    See Also:
    --------
    place, acker
    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb01bd
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb01bd'")

    # Convert the system inputs to NumPy arrays
    A_mat = np.array(A);
    B_mat = np.array(B);
    if (A_mat.shape[0] != A_mat.shape[1] or
        A_mat.shape[0] != B_mat.shape[0]):
        raise ControlDimension("matrix dimensions are incorrect")

    # Compute the system eigenvalues and convert poles to numpy array
    system_eigs = np.linalg.eig(A_mat)[0]
    placed_eigs = np.array(p);

    # SB01BD sets eigenvalues with real part less than alpha
    # We want to place all poles of the system => set alpha to minimum
    alpha = min(system_eigs.real);

    # Call SLICOT routine to place the eigenvalues
    A_z,w,nfp,nap,nup,F,Z = \
        sb01bd(B_mat.shape[0], B_mat.shape[1], len(placed_eigs), alpha,
               A_mat, B_mat, placed_eigs, 'C');

    # Return the gain matrix, with MATLAB gain convention
    return -F, nup
