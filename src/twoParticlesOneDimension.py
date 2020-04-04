from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import EulerChromer
from initpositions import SetPositions

# Simulate two particles in one dimension separated by a distance 1.5 sigma
solver = MDSolver(positions=SetPositions([[0.0], [1.5]]), 
                  T=5, 
                  dt=0.01)
solver(potential=LennardJones(solver, cutoff=3), 
       integrator=EulerChromer(solver),
       distance=True,
       dumpfile="../data/2N_1D_1.5S.data")
solver.plot_distance()
solver.plot_energy()

# Simulate two particles in one dimension separated by a distance 0.95 sigma
solver = MDSolver(positions=SetPositions([[0.0], [0.95]]), 
                  T=10, 
                  dt=0.01)
solver(potential=LennardJones(solver, cutoff=3), 
       integrator=EulerChromer(solver), 
       distance=True,
       dumpfile="../data/2N_1D_0.95S.data")
solver.plot_distance()
solver.plot_energy()
