""" Example: 864 particles in a box with periodic boundaries
Initial positions: Face-centered cube
Initial velocities: Temperatured-based
Time step: 0.01
Potential: Lennard-Jones
Integrator: Velocity-Verlet
"""
import matplotlib.pyplot as plt

from mdsolver import MDSolver
from mdsolver.analyze import Log
from mdsolver.initposition import FCC
from mdsolver.initvelocity import Temperature
from mdsolver.boundary import Periodic

solver = MDSolver(position=FCC(cells=6, lenbulk=10.2),
                  velocity=Temperature(T=2.5),
                  boundary=Periodic(lenbox=10.2),
                  dt=0.01)

# equilibration run
solver.thermo(10, "864N_3D_equi.log", "step", "time")
solver.run(steps=1000)
solver.snapshot("after_equi.xyz")

# production run
solver.dump(1, "864N_3D.xyz", "x", "y", "z")
solver.thermo(1, "864N_3D_prod.log", "step", "time", "temp", "poteng", "kineng", "velcorr", "mse")
solver.run(steps=1000, out="log")
solver.snapshot("final.xyz")

# analyze
logobj = Log("864N_3D_prod.log")
time = logobj.find("time")
temp = logobj.find("temp")
poteng = logobj.find("poteng")
kineng = logobj.find("kineng")
velcorr = logobj.find("velcorr")
mse = logobj.find("mse")

plt.figure()
plt.plot(time, temp)
plt.xlabel(r"Time, $t/\tau$")
plt.ylabel(r"Temperature, $T/T'$")

plt.figure()
plt.plot(time, velcorr)
plt.xlabel(r"Time, $t/\tau$")
plt.ylabel("Velocity-autocorrelation")

plt.figure()
plt.plot(time, mse)
plt.xlabel(r"Time, $t/\tau$")
plt.ylabel("Mean-squared displacement")

plt.figure()
plt.plot(time, poteng, label="Potential")
plt.plot(time, kineng, label="Kinetic")
plt.plot(time, poteng + kineng, label="Total")
plt.xlabel(r"Time, $t/\tau$")
plt.ylabel(r"Energy, $E/\varepsilon$")
plt.legend(loc='best')
plt.show()
