""" Example: 864 particles in a box with periodic boundaries
Initial positions: Face-centered cube
Initial velocities: Temperatured-based
Total time: 5 ps
Time step: 0.01 ps
Potential: Lennard-Jones
Integrator: Velocity-Verlet

Energy is plotted
"""

from mdsolver import MDSolver
from mdsolver.potential import LennardJones
from mdsolver.integrator import VelocityVerlet
from mdsolver.initpositions import FCC
from mdsolver.initvelocities import Temperature
from mdsolver.boundaryconditions import Periodic, Reflective

solver = MDSolver(positions=FCC(cells=6, lenbulk=10),
                  velocities=Temperature(T=1.5),
                  boundaries=Periodic(lenbox=12),
                  T=5,
                  dt=0.01)
solver.dump(10, "864N_3D.xyz", "x", "y", "z")
solver.thermo(10, "864N_3D.log", "step", "time", "poteng", "kineng")
solver.run(potential=LennardJones(solver, cutoff=3),
       integrator=VelocityVerlet(solver))
