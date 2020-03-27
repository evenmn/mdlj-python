from moleculardynamics import MDSolver
from potential import LennardJones

solver = MDSolver(positions='fcc', cells=6, lenbulk=10, T=5, dt=0.01)
solver.simulate(potential=LennardJones(cutoff=3), 
             integrator=solver.eulerChromer,
             poteng=True,
             dumpfile="../data/864N_3D.data")
solver.plot_energy()
