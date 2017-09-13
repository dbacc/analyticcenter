import numpy as np
from scipy import linalg
import control

from misc.misc import schur_complement, rsolve
import logging
import ipdb
from functools import partial


class DirectionAlgorithm(object):
    debug = True
    ac_object = None
    logger = logging.getLogger(__name__)
    system = None
    initial_X = None
    discrete_time = False
    maxiter = 100
    search_direction = None

    def __init__(self, ac_object, discrete_time):
        DirectionAlgorithm.ac_object = ac_object
        DirectionAlgorithm.system = ac_object.system
        if discrete_time:
            DirectionAlgorithm.initial_X = InitialXDT()
        else:
            DirectionAlgorithm.initial_X = InitialXCT()

        DirectionAlgorithm.discrete_time = discrete_time

    def _directional_iterative_algorithm(self, direction):
        def print_information(steps_count, residual, determinant, X):
            self.logger.info("Current step: {}\tResidual: {}".format(steps_count, residual))
            self.logger.debug("Current objective value (det(H(X))): {}".format(determinant))
            self.logger.debug("Current X:\n{}".format(X))
            det = np.real(determinant)
            if det < 0 :
                self.logger.critical("Something went wrong. Determinant ist negative. Aborting...")
                raise ValueError("Something went wrong. Determinant ist negative")

        X, success_init = self.initial_X()
        self.logger.debug("Initial X:\n{}".format(X))
        F, P = self.ac_object._get_F_and_P(X)

        determinant = linalg.det(P) * self.ac_object._get_determinant_R(X)
        A_F = (self.system.A - self.system.B @ F)
        steps_count = 0
        residual = self.ac_object.get_residual(X, P, F, A_F, self.search_direction)
        Delta_residual = float("inf")
        # ipdb.set_trace()
        while residual > self.ac_object.tol and Delta_residual > self.ac_object.rel_tol and steps_count < self.maxiter:
            # ipdb.set_trace()
            if self.debug:
                self.ac_object._get_H_matrix(X)
            ipdb.set_trace()
            print_information(steps_count, residual, determinant, X)
            R = self.ac_object._get_R(X)
            self.logger.debug("Current Determinant of R: {}".format(linalg.det(R)))
            Delta_X = direction(X, P, R, A_F)
            Delta_residual = linalg.norm(Delta_X)
            X = X + Delta_X
            self.logger.debug("Updating current X by Delta:_X:\n{}".format(Delta_X))

            F, P = self.ac_object._get_F_and_P(X)
            A_F = (self.system.A - self.system.B @ F)
            residual = self.ac_object.get_residual(X, P, F, A_F)
            determinant = linalg.det(P) * self.ac_object._get_determinant_R(X)
            steps_count += 1
        print_information(steps_count, residual, determinant, X)
        self.logger.info("Finished computation...")
        if residual <= self.ac_object.tol or Delta_residual <= self.ac_object.tol:
            self.ac_object.center = X
            self.ac_object.A_F = A_F
            return (X, True)
        else:

            return (X, False)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Please Implement this method")


class NewtonDirection(DirectionAlgorithm):
    def __init__(self):
        raise NotImplementedError("Should not be called from Base class")

    def __call__(self):
        self.logger.info("Computing Analytic Center with Newton approach")
        X, success =  self._directional_iterative_algorithm(
            direction=self._get_newton_direction)
        if success:
            self.logger.info("Computation of Analytic center with Newton approach was successful")
            self.logger.debug("At the analytic center A_F is:\n{}".format(self.ac_object.A_F))
            return X
        else:
            self.logger.critical("Computation of Analytic center was  not successful")
            return X


    def _get_newton_direction(self, X0, P0, R0, A_F):
        A_F_hat, P0_root, S2 = self._transform_system2current_X0(A_F, P0, R0)

        Delta_X_hat = self._newton_step_solver(A_F_hat, S2, P0_root)
        Delta_X = P0_root @ Delta_X_hat @ P0_root
        return Delta_X

    def _transform_system2current_X0(self, A_F, P0, R0):
        P0_root = linalg.sqrtm(P0)
        B_hat = P0_root @ self.system.B
        A_F_hat = rsolve(P0_root, (P0_root @ A_F))
        S2 = B_hat @ linalg.solve(R0, B_hat.H)  # only works in continuous time
        return A_F_hat, P0_root, S2

    def _splitting_method(self, A, S, P0_root):
        raise NotImplementedError


class NewtonDirectionMultipleDimensionsCT(NewtonDirection):
    def __init__(self):
        pass

    def _newton_step_solver(self, A, S, P0_root):
        '''solve newton step using kronecker products. Will be way too expensive in general!'''
        # TODO: use Schur Form
        rhs = np.ravel(A + A.H)
        AAH = A @ A.H
        n = self.system.n
        identity = np.identity(n)
        lhs = - np.kron(A.T, A) - np.kron(np.conj(A), A.H) - np.kron(identity, AAH) - np.kron(AAH.T,
                                                                                              identity) - np.kron(
            identity, S) - np.kron(S, identity)
        self.logger.debug("Current lhs and rhs\n{}\n{}".format(lhs, rhs))
        Delta = linalg.solve(lhs, rhs)
        self.logger.debug("Solution Delta:\n{}".format(Delta))
        Delta = np.reshape(Delta, [n, n])
        self.logger.debug("Reshaped Delta:\n{}".format(Delta))

        # check if indeed solution:



        if self.debug:
            self._check(A, S, Delta)
        return Delta

    def _check(self, A, S, Delta):
        AAH = A @ A.H
        n = self.system.n
        identity = np.identity(n)
        res = - A @ Delta @ A - AAH @ Delta - A.H @ Delta @ A.H - Delta @ AAH.H - S @ Delta - Delta @ S - A - A.H
        self.logger.debug("norm of the residual: {}".format(linalg.norm(res)))
        det_factor = linalg.det(identity - Delta @ A - A.H @ Delta - Delta @ S @ Delta)
        if det_factor < 1.:
            self.logger.critical("det factor by newton step is less than 1: {}".format(det_factor))
        else:
            self.logger.debug("det factor by newton step: {}".format(det_factor))

class NewtonDirectionIterativeCT(NewtonDirectionMultipleDimensionsCT):
    maxiter_newton = 200
    tol_newton = 10 ** -10

    def __init__(self):
        pass

    def _newton_step_solver(self, A, S, P0_root):
        alpha = 0  # TODO how to choose alpha?
        n = self.system.n
        identity = np.identity(n)
        AAH = A @ A.H
        API = A + identity
        Delta = 0 * A
        i = 0
        test = self.tol_newton + 1
        while i < self.maxiter_newton and linalg.norm(test) > self.tol_newton:
            rhs = 2 * alpha * Delta - A @ Delta @ A - A.H @ Delta @ A.H - A - A.H
            Delta = linalg.solve_lyapunov(AAH + S + alpha * identity, rhs)

            test = - A @ Delta @ A - AAH @ Delta - A.H @ Delta @ A.H - Delta @ AAH.H - S @ Delta - Delta @ S - A - A.H
            i += 1
            self.logger.debug(
                "Current value of Delta in Newton iteration:\n{}\nresidual: {}".format(Delta, linalg.norm(test)))
        return Delta


class NewtonDirectionMultipleDimensionsDT(NewtonDirection):
    def __init__(self):
        pass

    def _newton_step_solver(self, A, S, P0_root):
        '''solve newton step using kronecker products. Will be way too expensive in general!'''
        # TODO: use Schur Form

        AAH = A @ A.H
        n = self.system.n
        identity = np.identity(n)
        rhs = np.ravel(identity - S - AAH)
        lhs = np.kron(S, S) + np.kron(S, AAH) + np.kron(AAH.T, S) + np.kron(AAH.T, AAH) - np.kron(np.conj(A), A) \
              - np.kron(A.T, A.H) + np.kron(identity, identity)
        self.logger.debug("Current lhs and rhs\n{}\n{}".format(lhs, rhs))
        Delta = linalg.solve(lhs, rhs)
        self.logger.debug("Solution Delta:\n{}".format(Delta))
        Delta = np.reshape(Delta, [n, n])
        self.logger.debug("Reshaped Delta:\n{}".format(Delta))

        # check if indeed solution:
        # ipdb.set_trace()
        if self.debug:
            self._check(A, S, Delta)
        return Delta

    def _check(self, A, S, Delta):
        n = self.system.n
        identity = np.identity(n)
        AAH = A @ A.H
        res = S @ Delta @ S + AAH @ Delta @ S + S @ Delta @ AAH + AAH @ Delta @ AAH - A @ Delta @ A.H - A.H @ Delta @ A + Delta - identity + S + AAH
        self.logger.debug("norm of the residual: {}".format(linalg.norm(res)))

        det_factor = linalg.det(np.bmat([[identity - S @ Delta, A], [A.H @ Delta, identity + Delta]]))
        if det_factor < 1.:
            self.logger.critical("det factor by newton step is less than 1: {}".format(det_factor))
        else:
            self.logger.debug("det factor by newton step: {}".format(det_factor))


class NewtonDirectionOneDimensionCT(NewtonDirection):
    search_direction = None

    def __init__(self):
        pass

    # TODO: Check Formula: Slow convergence!!
    def _newton_step_solver(self, A, S, P0_root):
        search_dir = linalg.solve(P0_root, rsolve(P0_root, self.search_direction))
        self.logger.debug("Search direction: {}".format(self.search_direction))
        correction = -(np.real(np.trace(search_dir @ A + A.H @ search_dir)) / (np.real(
            np.trace(search_dir @ S @ search_dir)) + 0.5 * linalg.norm(
            search_dir @ A + A.H @ search_dir) ** 2))
        return correction * search_dir


class NewtonDirectionOneDimensionDT(NewtonDirection):
    search_direction = None

    def __init__(self):
        pass

    # TODO: Check Formula: Slow convergence!!
    def _newton_step_solver(self, A, S, P0_root):
        n = self.system.n
        identity = np.identity(n)
        AAH = A @ A.H
        search_dir = linalg.solve(P0_root, rsolve(P0_root, self.search_direction))
        second_order = - S @ search_dir @ S @ search_dir - 2 * AAH @ search_dir @ S @ search_dir - AAH @ search_dir @ AAH @ search_dir + 2 * A @ search_dir @ A.H @ search_dir - search_dir @ search_dir
        self.logger.debug("Search direction: {}".format(self.search_direction))
        return -0.5 * (np.real(np.trace(search_dir @ (identity - S - AAH))) / (np.real(
            np.trace(second_order)))) * search_dir


class SteepestAscentDirection(DirectionAlgorithm):
    def __init__(self, ac_object):
        pass

    def __call__(self):
        self.logger.info("Computing Analytic Center with steepest_ascent approach")
        self._directional_iterative_algorithm(direction=self._get_ascent_direction_direct)

    def _get_ascent_direction_direct(self, X, A_F):
        A_T = rsolve(self.riccati_operator(X), A_F)

        self.logger.debug("Current Feedback Matrix A_F:\n{}".format(A_F))
        self.logger.debug("Current Feedback Matrix transformed A_T:\n{}".format(A_T))
        A_T_symmetric = A_T + np.asmatrix(A_T).H
        # We're assuming simple eigenvalues here!
        largest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(self.system.n - 1, self.system.n - 1))
        smallest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(0, 0))
        self.logger.debug("Symmetric part of A_T:\n{}".format(A_T_symmetric))
        largest_eigenvector = largest_eigenpair[1]
        largest_eigenvalue = largest_eigenpair[0]

        smallest_eigenvector = smallest_eigenpair[1]
        smallest_eigenvalue = smallest_eigenpair[0]

        if np.abs(smallest_eigenvalue) < np.abs(largest_eigenvalue):
            largest_abs_eigenvalue = largest_eigenvalue
            smallest_abs_eigenvalue = smallest_eigenvalue
            largest_abs_eigenvector = largest_eigenvector
        else:
            largest_abs_eigenvalue = smallest_eigenvalue
            largest_abs_eigenvector = smallest_eigenvector
            smallest_abs_eigenvalue = largest_eigenvalue
        self.largest_eigenvalues = np.append(self.largest_eigenvalues, largest_abs_eigenvalue)
        self.smallest_eigenvalues = np.append(self.smallest_eigenvalues, smallest_abs_eigenvalue)
        Delta_X = largest_abs_eigenvector @ np.asmatrix(largest_abs_eigenvector).H

        self.logger.debug(
            "largest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(largest_eigenvalue,
                                                                                       largest_eigenvector,
                                                                                       linalg.norm(
                                                                                           largest_eigenvector)))
        self.logger.debug(
            "smallest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(smallest_eigenvalue,
                                                                                        smallest_eigenvector,
                                                                                        linalg.norm(
                                                                                            smallest_eigenvector)))

        stepsize = self._get_ascent_step_size(X, largest_abs_eigenvector)
        return stepsize * Delta_X

    def _get_ascent_direction_Ttransformed(self, X, A_F):
        T = linalg.sqrtm(self.riccati_operator(X))
        A_T = T @ rsolve(T, A_F)

        self.logger.debug("Current Feedback Matrix A_F:\n{}".format(A_F))
        self.logger.debug("Current Feedback Matrix transformed A_T:\n{}".format(A_T))

        A_T_symmetric = A_T + np.asmatrix(A_T).H
        # We're assuming simple eigenvalues here!
        largest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(self.system.n - 1, self.system.n - 1))
        smallest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(0, 0))

        self.logger.debug("Symmetric part of A_T:\n{}".format(A_T_symmetric))
        largest_eigenvector = largest_eigenpair[1]
        largest_eigenvalue = largest_eigenpair[0]
        smallest_eigenvector = smallest_eigenpair[1]
        smallest_eigenvalue = smallest_eigenpair[0]

        if np.abs(smallest_eigenvalue) < np.abs(largest_eigenvalue):
            largest_abs_eigenvalue = largest_eigenvalue
            largest_abs_eigenvector = largest_eigenvector
        else:
            largest_abs_eigenvalue = smallest_eigenvalue
            largest_abs_eigenvector = smallest_eigenvector
        Delta_X = T @ largest_abs_eigenvector @ np.asmatrix(largest_abs_eigenvector).H @ T

        self.logger.debug(
            "largest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(largest_eigenvalue,
                                                                                       largest_eigenvector,
                                                                                       linalg.norm(
                                                                                           largest_eigenvector)))
        self.logger.debug(
            "smallest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(smallest_eigenvalue,
                                                                                        smallest_eigenvector,
                                                                                        linalg.norm(
                                                                                            smallest_eigenvector)))
        stepsize = self._get_ascent_step_size(X, T @ largest_abs_eigenvector)
        return stepsize * Delta_X

    def _get_ascent_step_size(self, X, eigenvector):
        # perm = np.matrix([[0, 1], [1, 0]])
        H = self._get_H_matrix(X)
        eigenvector = np.asmatrix(eigenvector)  # / linalg.norm(eigenvector)
        self.logger.debug("H(X): {}".format(H))
        ProjectionMatrix = np.asmatrix(linalg.block_diag(eigenvector, eigenvector)).H @ np.bmat(
            [[self.system.A, self.system.B],
             [np.identity(self.system.n), np.zeros((self.system.n, self.system.m))]])

        ProjectedMatrix = ProjectionMatrix @ linalg.solve(H, ProjectionMatrix.H)
        c = ProjectedMatrix[1, 0]
        ab = ProjectedMatrix[0, 0] * ProjectedMatrix[1, 1]
        stepsize = c / (-ab + c ** 2)
        stepsize = c / (-ab + c ** 2)
        # ipdb.set_trace()
        self.logger.debug("Chosen stepsize: {}".format(stepsize))
        return stepsize


class InitialX(DirectionAlgorithm):
    X0 = None
    maxiter = 5

    def __init__(self):
        pass

    def __call__(self):
        if InitialX.X0 is None:
            self.logger.info('Computing initial X')

            X_plus = -self.riccati_solver(self.system.A, self.system.B, self.system.Q, self.system.R, self.system.S,
                                         np.identity(self.system.n))
            Am = -self.system.A
            Bm = self.system.B
            Sm = self.system.S
            Qm = -self.system.Q
            Rm = -self.system.R
            X_minus = -self.riccati_solver(Am, Bm, Qm, Rm, Sm,
                                          np.identity(self.system.n))

            if np.isclose(linalg.norm(X_plus-X_minus), 0):
                self.logger.critical("X_+ and X_- are (almost) identical: No interior!")
            self.logger.debug("Eigenvalues of X_plus: {}".format(linalg.eigh(X_plus)[0]))
            self.logger.debug("Eigenvalues of H(X_plus): {}".format(linalg.eigh(self.ac_object._get_H_matrix(X_plus))[0]))
            self.logger.debug("Eigenvalues of X_minus: {}".format(linalg.eigh(X_minus)[0]))
            self.logger.debug("Eigenvalues of H(X_minus): {}".format(linalg.eigh(self.ac_object._get_H_matrix(X_minus))[0]))
            if self.debug:
                self.ac_object._get_Hamiltonian()
            # ipdb.set_trace()
            newton_direction = self.newton_direction
            self.search_direction = 0.5 * (X_minus + X_plus)
            self.logger.debug("Eigenvalues of X_init_guess: {}".format(linalg.eigh(self.search_direction)[0]))
            self.logger.debug(
                "Eigenvalues of H(X_init_guess): {}".format(linalg.eigh(self.ac_object._get_H_matrix(self.search_direction))[0]))
            newton_direction.search_direction = self.search_direction
            InitialX.X0 = self.search_direction  # We use negative definite notion of solutions for Riccati equation
            self.logger.info("Improving Initial X with Newton approach")

            # ipdb.set_trace()
            Xinit, success = self._directional_iterative_algorithm(direction=newton_direction._get_newton_direction)

            if not success:
                self.logger.critical("Computation of initial X failed.")
            else:
                self.logger.debug("Eigenvalues of X_init: {}".format(linalg.eigh(Xinit)[0]))
                self.logger.debug(
                    "Eigenvalues of H(X_init): {}".format(linalg.eigh(self.ac_object._get_H_matrix(Xinit))[0]))

        else:
            self.logger.info("Initial X is already set")
            Xinit = self.X0
        return Xinit, True
class InitialXCT(InitialX):
    newton_direction = NewtonDirectionOneDimensionCT()
    riccati_solver = staticmethod(lambda *args: control.care(*args)[0])

    def __init__(self):
        pass


class InitialXDT(InitialX):
    newton_direction = NewtonDirectionOneDimensionDT()
    riccati_solver = staticmethod(lambda *args: control.dare(*args)[0])

    def __init__(self):
        pass
