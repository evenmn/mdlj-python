""" Example: A particle moving in two dimensions with reflective boundaries.
Initial position: (1,1)
Initial velocity: (1,1)
Length of box: 2
Total time: 10 ps
Time step: 0.01 ps
Potential: Lennard-Jones
Integrator: Forward-Euler

The position is plotted as a function of time, energy is not plotted
"""

from moleculardynamics import MDSolver
from potential import LennardJones
from integrator import ForwardEuler
from initpositions import SetPositions
from initvelocities import SetVelocities
from boundaryconditions import Reflective

# Simulate two particles in one dimension separated by a distance 1.5 sigma
solver = MDSolver(positions=SetPositions([[1.0, 1.0]]), 
                  velocities=SetVelocities([[1.0, 1.0]]),
                  boundaries=Reflective(lenbox=2), 
                  T=10, 
                  dt=0.01)
solver(potential=LennardJones(solver), 
       integrator=ForwardEuler(solver),
       poteng=False)

# Plot the position as a function of time
from numpy import linspace
from matplotlib.pyplot import plot, show, xlabel, ylabel
r = solver.r.flatten()
plot(linspace(0,10,len(r)), r)
xlabel("Time [ps]")
ylabel("Position")
show()
