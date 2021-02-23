import time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MDSolver:
    """ Initialize the MDSolver class. This includes defining the
    time scales, initialize positions and velocities, and define
    matplotlib fixes.

    Parameters
    ----------
    positions : obj
        class object defined by initpositions.py. Face-centered cube
        with length 3 and 4 particles as default.
    velocity : obj
        class object defined by initvelocities.py. No velocity as default.
    boundaries : obj
        class object defined by boundaryconditions.py. Open boundaries
        as default.
    T : float
        total time
    dt : float
        time step
    """

    from .dump import Dump
    from .thermo import Thermo
    from .initpositions import FCC
    from .initvelocities import Zero
    from .boundaryconditions import Open
    from .integrator import VelocityVerlet
    from .potential import LennardJones

    def __init__(self, dt, positions, velocities=Zero(), boundaries=Open()):

        self.t = 0
        self.dt = dt

        # Initialize positions
        self.r = positions()

        # Initialize velocities
        self.v = velocities(self.r.shape)

        self.numparticles, self.numdimensions = self.r.shape

        # Set objects
        self.compute_poteng = False
        self.boundaries = boundaries
        self.integrator = self.VelocityVerlet(self)
        self.potential = self.LennardJones(self, cutoff=3)

        # Initialize acceleration
        self.a, _ = self.potential(self.r)

        # Initialize the number of times each particle has touched the wall
        self.n = np.zeros(self.r.shape, int)

        # print to terminal
        self.print_to_terminal()

        self.dump(np.inf, "file.dump")
        self.thermo(np.inf, "file.thermo")

    def __repr__(self):
        return "MDSolver base class"

    def set_potential(self, potential):
        """Set force-field
        """
        self.potential = potential

    def set_integrator(self, integrator):
        """Set integrator
        """
        self.integrator = integrator

    def dump(self, freq, file, *quantities):
        """Dump atom-quantities to file
        """
        self.dumpobj = self.Dump(freq, file, quantities)

    def thermo(self, freq, file, *quantities):
        """Print thermo-quantities to file
        """
        if "poteng" in quantities:
            self.compute_poteng = True
        else:
            self.compute_poteng = False
        self.thermoobj = self.Thermo(freq, file, quantities)

    def snapshot(self, filename):
        """Take snapshot of system and write to xyz-file
        """
        lst = ('x', 'y', 'z')
        tmp_dumpobj = self.Dump(1, filename, lst[:self.numdimensions])
        tmp_dumpobj(self)
        del tmp_dumpobj

    def print_to_terminal(self):
        """ Print information to terminal
        """
        print("\n\n" + 14 * "=", " SYSTEM INFORMATION ", 14 * "=")
        print("Number of particles:  ", self.numparticles)
        print("Number of dimensions: ", self.numdimensions)
        print("")
        print("Potential:            ", self.potential)
        print("Boundary conditions:  ", self.boundaries)
        print("Integrator:           ", self.integrator)
        print("Timestep:             ", self.dt)
        print(50 * "=" + "\n\n")

    def run(self, steps):
        """ Integration loop. Computes the time-development of position and
        velocity using a given integrator and inter-atomic potential.

        Parameters
        ----------
        potential : obj
            object defining the inter-atomic potential
        integrator : obj
            object defining the integrator
        """
        self.t0 = self.t
        # Integration loop
        start = time.time()
        while self.t < self.t0 + steps + 1:
            self.r, n, self.v, self.a, self.u = self.integrator(self.r, self.v, self.a)
            self.n += n

            self.dumpobj(self)
            self.thermoobj(self)

            self.t += 1
        end = time.time()
        print("Elapsed time: ", end-start)

    def __del__(self):
        del self.dumpobj, self.thermoobj
