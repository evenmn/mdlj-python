""" Example: 864 particles in a box with periodic boundaries
Initial positions: Face-centered cube
Initial velocities: Temperatured-based
Total time: 10 ps
Time step: 0.01 ps
Potential: Lennard-Jones
Integrator: Velocity-Verlet

Energy, temperature and mean square displacement are plotted
"""

from mdsolver import MDSolver
from mdsolver.potential import LennardJones
from mdsolver.integrator import VelocityVerlet
from mdsolver.initpositions import FCC
from mdsolver.initvelocities import Temperature
from mdsolver.boundaryconditions import Periodic, Reflective

solver = MDSolver(positions=FCC(cells=6, lenbulk=10), 
                  velocities=Temperature(T=300),
                  boundaries=Periodic(lenbox=10),
                  T=10, 
                  dt=0.01)
solver(potential=LennardJones(solver, cutoff=3), 
       integrator=VelocityVerlet(solver),
       msd=True,
       dumpfile="864N_3D.data")
solver.plot_energy()
solver.plot_temperature()
solver.plot_msd()
