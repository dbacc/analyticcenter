# Analytic Center

This python package is intended to be used for the computation of the analytic center of a strictly passive and stable discrete-time or continuous-time linear time-invariant system.
One can choose between several solvers (Newton, steepest ascent). Note that this package is a proof-of-concept implementation. There is no guaranty for robustness or optimization of the code.

## Installation
The package can be installed by running

`python setup.py install`


Note that you need at least python version 3.5 to use this software.

Alternatively, you can install the `conda` package by invoking
`conda install -c dbankmann analyticcenter`



## Example usage
### Simple example
The following simple example can be found in `examples/example3.py`.
Run the example with `python3 -O examples/example3.py`. Ommiting the `-O` switch will turn on some debugging information.
* First one has to create a system object `sys`:

```python
import numpy as np

from analyticcenter import WeightedSystem

A = np.matrix([[1, -1], [1, 1]])
B = np.matrix([[1], [1]])
C = B.T
D = C @ B
Q = np.zeros((2,2))
sys = WeightedSystem(A, B, C, D, Q, B, D + D.T)
```

* Then one has to create an algorithm object, where you have to define the type of the system (discrete, continuous) and you can optionally provide some tolerances.

```python
from analyticcenter import get_algorithm_object

from analyticcenter.examples.example2 import sys

if __name__ == "__main__":

    alg = get_algorithm_object(sys, 'newton', discrete_time=False, save_intermediate=True)
    alg()
```

* The resulting _analytic center_ object `ac` contains data and methods for analyzing the system at the analytic center.

### Jupyter notebooks
There are a few jupyter notebooks in the examples folder showing all the results (and more) that are used in the references below.



## Tests
Some tests for the basic functionality have been written. You can run those with 

`python setup.py test`

## Logging
Uses pythons `logging` module. Configuration can be found in `config/config.yaml`. If the package should be used as a library, the automatic configuration should be disabled by uncommenting the appropriate line in `__init__.py`.
## License
See the provided license file.

## Author
    Daniel Bankmann
    Technische Universität Berlin
    bankmann@math.tu-berlin.de

## References
	Bankmann, D.; Mehrmann, V.; Nesterov, Y.; van Dooren, P., "Computation of the analytic center of the solution set of the linear matrix inequality arising in continuous- and discrete-time passivity analysis, 2019"


	C. A. Beattie, V. Mehrmann, and P. Van Dooren, “Robust port-Hamiltonian representations of passive systems,” Automatica, vol. 100, pp. 182–186, Feb. 2019.
