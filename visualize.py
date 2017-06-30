import matplotlib.pyplot as plt
import numpy as np
def plot_eigenvalues(alg_object, n = 20):
    plt.figure()
    plt.plot(np.abs(alg_object.largest_eigenvalues)[:n], 'x')
    plt.plot(np.abs(alg_object.smallest_eigenvalues)[:n], 'o')
    plt.plot(np.sqrt(np.abs(alg_object.largest_eigenvalues)[:n]* np.abs(alg_object.smallest_eigenvalues)[:n]), 'H')
    plt.show()

def log_plot_eigenvalues(alg_object, n=100):
    plt.figure()
    plt.loglog(np.abs(alg_object.largest_eigenvalues)[:n], 'x')
    plt.loglog(np.abs(alg_object.smallest_eigenvalues)[:n], 'o')
    plt.loglog(np.sqrt(np.abs(alg_object.largest_eigenvalues)[:n]* np.abs(alg_object.smallest_eigenvalues)[:n]), 'h')
    plt.show()
