from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import VelocityVerlet

solver = MDSolver(positions='fcc', cells=4, lenbox=10, T=5, dt=0.01)
solver(potential=LennardJones(solver), 
       integrator=VelocityVerlet(solver), 
       dumpfile="../data/256N_3D.data")
solver.plot_energy()
