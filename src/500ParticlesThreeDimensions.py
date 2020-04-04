from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import VelocityVerlet
from initpositions import FCC

solver = MDSolver(positions=FCC(cells=5, lenbulk=10), 
                  T=5, 
                  dt=0.01)
solver(potential=LennardJones(solver), 
       integrator=VelocityVerlet(solver), 
       dumpfile="../data/500N_3D.data")
solver.plot_energy()
