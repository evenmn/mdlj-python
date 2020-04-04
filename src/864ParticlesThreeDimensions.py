from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import VelocityVerlet
from initpositions import FCC
from initvelocities import Temperature

solver = MDSolver(positions=FCC(cells=6, lenbulk=10), 
                  velocities=Temperature(T=300),
                  T=5, 
                  dt=0.01)
solver(potential=LennardJones(solver), 
       integrator=VelocityVerlet(solver),
       dumpfile="../data/864N_3D.data")
solver.plot_energy()
solver.plot_temperature()
