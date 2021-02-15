""" Example: 864 particles in a box with periodic boundaries
Initial positions: Face-centered cube
Initial velocities: Temperatured-based
Time step: 0.01 ps
Potential: Lennard-Jones
Integrator: Velocity-Verlet
"""

from mdsolver import MDSolver
from mdsolver.initpositions import FCC, Restart
from mdsolver.initvelocities import Temperature
from mdsolver.boundaryconditions import Periodic

solver = MDSolver(positions=FCC(cells=6, lenbulk=10),
                  velocities=Temperature(T=1.5),
                  boundaries=Periodic(lenbox=12),
                  dt=0.01)

# equilibration run
solver.thermo(10, "864N_3D.log", "step", "time")
solver.run(steps=100)

# production run
solver.dump(10, "864N_3D.xyz", "x", "y", "z")
solver.thermo(10, "864N_3D.log", "step", "time", "poteng", "kineng")
solver.run(steps=100)
solver.snapshot("snapshot.xyz")
