##
## Copyright (c) 2019
##
## @author: Daniel Bankmann
## @company: Technische Universität Berlin
##
## This file is part of the python package analyticcenter
## (see https://gitlab.tu-berlin.de/PassivityRadius/analyticcenter/)
##
## License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
##
import matplotlib.pyplot as plt
import numpy as np


def plot_eigenvalues(alg_object, n=20):
    plt.figure()
    plt.plot(np.abs(alg_object.largest_eigenvalues)[:n], 'x')
    plt.plot(np.abs(alg_object.smallest_eigenvalues)[:n], 'o')
    plt.plot(np.sqrt(np.abs(alg_object.largest_eigenvalues)[:n] * np.abs(alg_object.smallest_eigenvalues)[:n]), 'H')
    plt.show()


def log_plot_eigenvalues(alg_object, n=100):
    plt.figure()
    plt.loglog(np.abs(alg_object.largest_eigenvalues)[:n], 'x')
    plt.loglog(np.abs(alg_object.smallest_eigenvalues)[:n], 'o')
    plt.loglog(np.sqrt(np.abs(alg_object.largest_eigenvalues)[:n] * np.abs(alg_object.smallest_eigenvalues)[:n]), 'h')
    plt.show()


def log_log_direction(X, det, X_final=None, det_final=None):
    X = np.array(X)
    det = np.array(det)
    if X_final is None:
        X_final = X[-1]
    if det_final is None:
        det_final = det[-1]
    X_diff = np.linalg.norm( (X - X_final) / X_final, axis=(1,2))[:-1]
    det_diff = np.abs(det - det_final) / det_final
    t = np.arange(len(det_diff))
    t = t.astype('float')
    plt.figure()
    plt.yscale('log')
    plt.xlabel('k')

    plt.plot(X_diff, label=r'$\|\|X-X_c\|\|/ \|\|X_c\|\|$')
    plt.legend()
    plt.savefig('figure1.pdf')
    plt.figure()
    plt.yscale('log')
    plt.xlabel('k')
    plt.plot(det_diff, label=r'$\|det(W(X))-det(W(X_c))\|/ det(W(X_c))$')
    plt.legend()
    plt.savefig('figure2.pdf')
    plt.show()

