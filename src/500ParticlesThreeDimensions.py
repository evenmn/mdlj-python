from moleculardynamics import MDSolver
from potential import LennardJones

solver = MDSolver(positions='fcc', cells=5, lenbulk=10, T=5, dt=0.01)
solver.simulate(potential=LennardJones(cutoff=3), 
                integrator=solver.velocityVerlet, 
                poteng=True,
                dumpfile="../data/500N_3D.data")
solver.plot_energy()
