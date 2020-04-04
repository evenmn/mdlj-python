""" Example: Two particles moving in one dimension with open boundaries
Initial positions: ((0), (1.5))
Initial velocities: ((0), (0))
Total time: 10 ps
Time step: 0.01 ps
Potential: Lennard-Jones
Integrator: Euler-Chromer

The distance between the particles and the energy is plotted
"""

from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import EulerChromer
from initpositions import SetPositions

# Simulate two particles in one dimension separated by a distance 1.5 sigma
solver = MDSolver(positions=SetPositions([[0.0], [1.5]]), 
                  T=5, 
                  dt=0.001)
solver(potential=LennardJones(solver), 
       integrator=EulerChromer(solver),
       distance=True,
       dumpfile="../data/2N_1D_1.5S.data")
solver.plot_distance()
solver.plot_energy()
