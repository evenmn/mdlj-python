# Python-MD
Molecular dynamics solver with the Lennard-Jones potential written in object-oriented Python for teaching purposes. The solver supports various integrators, boundary condition and initialization methods. However, it is simple as it only supports the Lennard-Jones potential and the microcanonical ensemble (NVE). Also, it only supports symmetric systems, i.e., all sides have the same length and take the same boundary conditions. Albeit efforts are put in making the code fast (mostly by replacing loops by vectorized operations), the performance cannot compete with packages written in low-level languages. 

## Installation
First download the contents:
``` bash
$ git clone https://github.com/evenmn/Python-MD
```
and then install the mdsolver:
``` bash
$ cd Python-MD
$ pip install .
```

## Example usage
A example script could look like this:
``` python
from mdsolver import MDSolver
from mdsolver.initpositions import SetPosition

solver = MDSolver(positions=SetPosition([[0.0], [1.5]]), dt=0.01)
solver.thermo(1, "log.mdsolver", "step", "time", "poteng", "kineng")
solver.run(steps=1000)
```

For more examples, see the examples folder.
