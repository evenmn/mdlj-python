from moleculardynamics import MDSolver
from potential import LennardJones

# Simulate two particles in one dimension separated by a distance 1.5 sigma
solver = MDSolver(positions=[[0.0], [1.5]], T=5, dt=0.01)
solver.simulate(potential=LennardJones(cutoff=3), 
                integrator=solver.velocityVerlet, 
                distance=True,
                poteng=True,
                dumpfile="../data/2N_1D_1.5S.data")
solver.plot_distance()
solver.plot_energy()

# Simulate two particles in one dimension separated by a distance 0.95 sigma
solver = MDSolver(positions=[[0.0], [0.95]], T=5, dt=0.01)
solver.simulate(potential=LennardJones(cutoff=3), 
                integrator=solver.eulerChromer, 
                distance=True,
                poteng=True,
                dumpfile="../data/2N_1D_0.95S.data")
solver.plot_distance()
solver.plot_energy()
