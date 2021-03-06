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

## Example: Two oscillating particles in one dimension
A simple example where two particles interact with periodic motion can be implemented like this:
``` python
from mdsolver import MDSolver
from mdsolver.initpositions import SetPosition

solver = MDSolver(positions=SetPosition([[0.0], [1.5]]), dt=0.01)
solver.thermo(1, "log.mdsolver", "step", "time", "poteng", "kineng")
solver.run(steps=1000)
```

## Example: 864 particles in three dimensions with PBC
A more complex example where 6x6x6x4=864 particles in three dimensions interact and where the boundaries are periodic is shown below. The particles are initialized in a face-centered cube, and the initial temperature is 300K (2.5 in Lennard-Jones units). We first perform an equilibration run, and then a production run.
``` python
from mdsolver import MDSolver
from mdsolver.initpositions import FCC
from mdsolver.initvelocities import Temperature
from mdsolver.boundaryconditions import Periodic

solver = MDSolver(positions=FCC(cells=6, lenbulk=10),
                  velocities=Temperature(T=2.5),
                  boundaries=Periodic(lenbox=12),
                  dt=0.01)

# equilibration run
solver.thermo(10, "equilibration.log", "step", "time")
solver.run(steps=1000)
solver.snapshot("after_equi.xyz")

# production run
solver.dump(1, "864N_3D.xyz", "x", "y", "z")
solver.thermo(1, "production.log", "step", "time", "temp", "poteng", "kineng")
solver.run(steps=1000, out="log")
solver.snapshot("final.xyz")
```

## Post-process simulations
The thermo style outputs (temperature, energy etc...) are stored in a log file, rather than in arrays. This has two purposes: Storing thermo style outputs in arrays might be memory intensive, and the file can be kept for later simulations. Reading these log files (here `production.log`) can easily be done using the Log-class:
``` python
from mdsolver.analyze import Log
logobj = Log("production.log")
time = logobj.find("time")
temp = logobj.find("temp")
```
The `find`-method outputs a numpy array.

For more examples, see the examples folder.
