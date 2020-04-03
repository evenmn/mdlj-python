from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import EulerChromer
from initpositions import setPositions

solver = MDSolver(positions=setPositions([[0, 0], [2, 0], [0, 2], [1.5, 1.5]]),
               T=5, 
               dt=0.01)
solver(potential=LennardJones(solver),
       integrator=EulerChromer(solver),
       distance=True,
       dumpfile="../data/4N_2D.data")
solver.plot_distance()
solver.plot_energy()
