""" Example: A particle moving in two dimensions with reflective boundaries.
Initial position: (1,1)
Initial velocity: (1,1)
Length of box: 2
Total time: 10 ps
Time step: 0.01 ps
Potential: Lennard-Jones
Integrator: Forward-Euler

The position is plotted as a function of time, energy is not plotted
"""

from mdsolver import MDSolver
from mdsolver.initpositions import SetPosition
from mdsolver.initvelocities import SetVelocity
from mdsolver.boundaryconditions import Reflective

# Simulate two particles in one dimension separated by a distance 1.5 sigma
solver = MDSolver(positions=SetPosition([[1.0, 1.0]]), 
                  velocities=SetVelocity([[1.0, 1.0]]),
                  boundaries=Reflective(lenbox=2), 
                  dt=0.01)
solver.dump(1, "1N_2D.xyz", "x", "y")
solver.run(steps=1000)
