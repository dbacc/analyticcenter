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


def log_log_direction(X, det, k_last=6):
    X = np.array(X)
    det = np.array(det)
    X_final = X[-1]
    det_final = det[-1]
    X_diff = np.linalg.norm(X - X_final, axis=(1,2))[:-1]
    det_diff = np.abs(det - det_final)[:-1]
    t = np.arange(len(det_diff))
    t = t.astype('float')[-k_last:]
    quadratic_curve_det = np.polyfit(t, np.log(det_diff[-k_last:]), 2)
    quadratic_curve_det = np.polyval(quadratic_curve_det, t)
    quadratic_curve_X = np.polyfit(t, np.log(X_diff[-k_last:]), 2)
    quadratic_curve_X = np.polyval(quadratic_curve_X, t)
    plt.figure()
    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('n')

    plt.plot(X_diff, label=r'$\|\|X-X_c\|\|$')
    # plt.plot(t, np.exp(quadratic_curve_X), label='best quadratic fit')
    plt.legend()
    quadratic_curve = 2*np.linalg.norm(X_diff[0])-t**2
    plt.figure()
    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('n')
    plt.plot(det_diff, label=r'$\|det(H(X))-det(H(X_c))\|$')
    # plt.plot(t, np.exp(quadratic_curve_det),label='best quadratic fit')
    plt.legend()
    plt.show()