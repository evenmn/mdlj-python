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
from mdsolver.potential import LennardJones
from mdsolver.integrator import EulerChromer
from mdsolver.initpositions import SetPositions
from mdsolver.tasks import PlotEnergy, DumpPositions, PlotDistance

solver = MDSolver(positions=SetPositions([[0.0], [1.5]]), 
                  T=5, 
                  dt=0.01)
solver(potential=LennardJones(solver), 
       integrator=EulerChromer(solver),
       tasks=[PlotEnergy(solver),
              DumpPositions(solver, "2N_1D_1.5S.data"),
              PlotDistance(solver)])
