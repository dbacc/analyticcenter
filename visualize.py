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


def log_log_direction(X, det):
    X_final = X[-1]
    det_final = det[-1]
    X_diff = X - X_final
    det_diff = det - det_final
    t = np.arange(len(det))
    t = t.astype('float')
    quadratic_curve = np.abs(det_diff[0])*t**(-2)
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('n')

    plt.plot(np.linalg.norm(X_diff, axis=(1,2)), label=r'$\|\|X-X_c\|\|$')
    plt.plot(t, quadratic_curve, label='quadratic function')
    plt.legend()
    quadratic_curve = np.linalg.norm(X_diff[0])*t**(-2)
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('n')
    plt.plot(np.abs(det_diff), label=r'$\|det(H(X))-det(H(X_c))\|$')
    plt.plot(t, quadratic_curve,label='quadratic function')
    plt.legend()
    plt.show()