""" Example: Two particles moving in one dimension with open boundaries
Initial positions: ((0), (1.5))
Initial velocities: ((0), (0))
Total time: 10 ps
Time step: 0.01 ps
Potential: Lennard-Jones
Integrator: Euler-Chromer

The distance between the particles and the energy is plotted
"""

from mdsolver import MDSolver
from mdsolver.initposition import SetPosition

# Simulate two particles in one dimension separated by a distance 1.5 sigma
position = SetPosition([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5], [1.5, 1.5]])
solver = MDSolver(position=position, dt=0.01)
solver.dump(1, "4N_2D.xyz", "x", "y")
solver.run(steps=1000)
