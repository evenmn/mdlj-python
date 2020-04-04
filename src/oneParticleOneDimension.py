from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import EulerChromer
from initpositions import SetPositions
from boundaryconditions import Reflective

# Simulate two particles in one dimension separated by a distance 1.5 sigma
solver = MDSolver(positions=SetPositions([[0.0]]), 
                  boundaries=Reflective(lenbox=10), 
                  T=100, 
                  dt=0.01)
solver(potential=LennardJones(solver, cutoff=3), 
       integrator=EulerChromer(solver),
       dumpfile="../data/1N_1D.data")
solver.plot_energy()
