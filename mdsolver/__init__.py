import sys
import time
import tqdm
import datetime
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
    from .initposition import FCC
    from .initvelocity import Zero
    from .boundary import Open
    from .integrator import VelocityVerlet
    from .potential import LennardJones

    def __init__(self, dt, position, velocity=Zero(), boundary=Open(), info=True):

        self.t = 0
        self.dt = dt

        # Initialize position
        self.r = position()

        # Initialize velocity
        self.v = velocity(self.r.shape)

        self.numparticles, self.numdimensions = self.r.shape

        # Set objects
        self.compute_poteng = False
        self.boundary = boundary
        self.integrator = self.VelocityVerlet(self)
        self.potential = self.LennardJones(self, cutoff=3)

        # Initialize acceleration
        self.a, _ = self.potential(self.r)

        # Initialize the number of times each particle has touched the wall
        self.n = np.zeros_like(self.r)

        # print to terminal
        self.info = info
        if self.info:
            self.print_to_terminal()

        self.dumpobj = self.Dump(np.inf, "dump.xyz", ())
        self.thermoobj = self.Thermo(np.inf, "log.mdsolver", ())

    def __repr__(self):
        return "MDSolver base class"

    def print_to_terminal(self):
        """ Print information to terminal
        """
        now = datetime.datetime.now()
        print(f"Simulation started on {now:%Y-%m-%d %H:%M:%S}")
        print("\n" + 14 * "=", " SYSTEM INFORMATION ", 14 * "=")
        print("Number of particles:  ", self.numparticles)
        print("Number of dimensions: ", self.numdimensions)
        print("")
        print("Potential:            ", self.potential)
        print("Boundary conditions:  ", self.boundary)
        print("Integrator:           ", self.integrator)
        print("Timestep:             ", self.dt)
        print(50 * "=")

    def set_potential(self, potential):
        """Set force-field
        """
        if self.info:
            print("\nPotential changed, new potential: ", str(potential))
        self.potential = potential

    def set_integrator(self, integrator):
        """Set integrator
        """
        if self.info:
            print("\nIntegrator changed, new integrator: ", str(integrator))
        self.integrator = integrator

    def dump(self, freq, file, *quantities):
        """Dump atom-quantities to file
        """
        if self.info:
            print(f"\nDumping every {freq}th (", ", ".join(quantities), f") to file '{file}'")
        self.dumpobj = self.Dump(freq, file, quantities)

    def thermo(self, freq, file, *quantities):
        """Print thermo-quantities to file
        """
        if self.info:
            print(f"\nPrinting every {freq}th (", ", ".join(quantities), f") to file '{file}'")
        if "poteng" in quantities:
            self.compute_poteng = True
        else:
            self.compute_poteng = False
        self.thermoobj = self.Thermo(freq, file, quantities)

    def snapshot(self, filename, vel=False):
        """Take snapshot of system and write to xyz-file
        """
        if self.info:
            print(f"\nSnapshot saved to file '{filename}'")
        if vel:
            lst = ('x', 'y', 'z', 'vx', 'vy', 'vz')
        else:
            lst = ('x', 'y', 'z')
        tmp_dumpobj = self.Dump(1, filename, lst[:self.numdimensions])
        tmp_dumpobj(self)
        del tmp_dumpobj

    def write_rdf(self, filename, max_radius, nbins="auto"):
        """Radial distribution function (RDF)
        """
        if self.info:
            print(f"\nWriting radial distribution function to file '{filename}'")
            print(f"Max radius: {max_radius}. Number of bins: {nbins}")

        # volume computation
        min_ = np.min(self.r, axis=0)
        max_ = np.max(self.r, axis=0)
        length = max_ - min_
        volume = np.prod(length)
    
        # compute distance between all particles (with PBC)
        x, y = self.r[:, np.newaxis, :], self.r[np.newaxis, :, :]
        dr = x - y
        dr = self.boundary.checkDistance(dr)
        drNorm = np.linalg.norm(dr, axis=2).flatten()
    
        # count number of distances within each bin
        n, bin_edges = np.histogram(drNorm, bins=nbins, range=(0, max_radius))
        n[0] = 0
    
        # find bin centeres and bin widths
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_sizes = bin_edges[1:] - bin_edges[:-1]

        # normalize 
        norm = [1, 2*np.pi*bin_centres, 4*np.pi*bin_centres**2]
        rdf = (volume / self.numparticles**2) * n / (norm[self.numdimensions-1]*bin_sizes)
        np.savetxt(filename, [bin_centres, rdf])

    def run(self, steps, out="tqdm"):
        """ Integration loop. Computes the time-development of position and
        velocity using a given integrator and inter-atomic potential.

        Parameters
        ----------
        potential : obj
            object defining the inter-atomic potential
        integrator : obj
            object defining the integrator
        """
        if self.info:
            print(f"\nRunning {steps} time steps. Output mode {out}")
        self.t0 = self.t

        # Integration loop
        iterations = range(self.t0, self.t0 + steps + 1)
        if out == "tqdm":
            sys.stdout.flush()
            iterations = tqdm.tqdm(iterations)
        elif out == "log":
            self.thermoobj.write_header()
        # else: whatever else will give no output ("no", "off", "false" etc)

        start = time.time()
        for self.t in iterations:
            self.r, n, self.v, self.a, self.u = self.integrator(self.r, self.v, self.a)
            self.n += n

            self.dumpobj(self)
            log = self.thermoobj(self)
            if out == "log":
                print(log, end="")

            self.t += 1
        end = time.time()
        if out == "log":
            print("Elapsed time: ", end-start)
        self.dumpobj.f.flush()
        self.thermoobj.f.flush()

    def __del__(self):
        del self.dumpobj, self.thermoobj
