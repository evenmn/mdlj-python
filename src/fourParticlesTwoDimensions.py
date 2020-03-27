from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import EulerChromer

solver = MDSolver(positions=[[0, 0], [2, 0], [0, 2], [1.5, 1.5]],
               T=5, 
               dt=0.01)
solver.simulate(potential=LennardJones(cutoff=3), 
                integrator=EulerChromer(solver), 
                distance=True,
                dumpfile="../data/4N_2D.data")
solver.plot_distance()
solver.plot_energy()
