from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import VelocityVerlet

solver = MDSolver(positions='fcc', cells=6, lenbox=10, T=5, dt=0.01)
solver.simulate(potential=LennardJones(cutoff=3), 
                integrator=VelocityVerlet(solver),
                dumpfile="../data/864N_3D.data")
solver.plot_energy()
