# Analytic Center

This python package is intended to be used for the computation of the analytic center of a strictly passive and stable discrete-time or continuous-time linear time-invariant system.
One can choose between several solvers (Newton, steepest ascent).

## Installation
The package can be installed by running

`python setup.py install`

The package requirements are listed in `requirements.txt` and can be installed with 

`pip install -r requirements.txt`

Note that you need at least python version 3.5 to use this software.


## Example usage
The following simple example can be found in `examples/example3.py`.
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
from analyticcenter import NewtonDirectionMultipleDimensionsCT
from analyticcenter.examples.example3 import sys

if __name__ == "__main__":
    alg = get_algorithm_object(sys, discrete_time=False, save_intermediate=True, abs_tol=9e-12)
    direction_method1 = NewtonDirectionMultipleDimensionsCT()
    direction_method1.maxiter = 40
    (ac, success) = direction_method1()
```
* The resulting _analytic center_ object `ac` contains data and methods for analyzing the system at the analytic center.

## Tests
Some tests for the basic functionality have been written. You can run those with 

`python setup.py test`

## Logging
Uses pythons `logging` module. Configuration can be found in `config/config.yaml`. If the package should be used as a library, the automatic configuration should be disabled by uncommenting the appropriate line in `__init__.py`.
## License
See the provided license file.

## Author
Daniel Bankmann\
Technische Universität Berlin\
bankmann@math.tu-berlin.de
