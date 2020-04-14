import numpy as np
import matplotlib.pyplot as plt

# for plotting
label_size = {"size":14}                # Dictionary with size
plt.style.use("bmh")                    # Beautiful plots
plt.rcParams["font.family"] = "Serif"   # Font

class Tasks:
    """ Which physical observables to display
    """
    def __init__(self, solver):
        self.solver = solver
        
    #@classmethod
    def _update(self, state, step):
       """ Update task every timestep
       """
       raise NotImplementedError("Class {} has no instance 'update'."
                                  .format(self.__class__.__name__))
       
    def __call__(self):
       """ Perform task
       """
       raise NotImplementedError("Class {} has no instance '__call__'."
                                  .format(self.__class__.__name__))
        
class DumpPositions(Tasks):
    """ Dump positions 
    """
    def __init__(self, solver, dumpfile):
        self.solver=solver
        self.dumpfile = dumpfile
        self.f = open(dumpfile,'w')       # Open dumpfile
        
    def __str__(self):
        return "Dumping position to the file {}.".format(self.dumpfile)
        
    @staticmethod
    def dumpPositions(r, dumpfile):
        """ Dumping positions at timestep t to a dumpfile. We use the xyz-
        format, which can easily be visualized using Ovito.
        
        Parameters
        ----------
        r : ndarray
            position array
        dumpfile : str
            name and address of dumpfile
        """
        numparticles = len(r)
        dat = np.column_stack((numparticles * ['Ar'], r))
        np.savetxt(dumpfile, dat, header="{}\ntype x y z"
                   .format(numparticles), fmt="%s", comments='')
        
    #@classmethod
    def _update(self, state, step):
        self.dumpPositions(state.r, self.f)
        
    def __call__(self):
        self.f.close()

class PlotPositions(Tasks):
    """ Plot position
    """
    def __init__(self, solver):
        self.solver = solver
        self.time = solver.time
        self.r = np.zeros((solver.N, solver.numparticles, solver.numdimensions))
    
    def __str__(self):
        return "Plotting positions as a function of time."
    
    def _update(self, state, step):
        """ Append to position array.
        """
        self.r[step] = state.r
        
    def __call__(self):
        """ Plot position as a function of time.
        """
        r = self.r[:,0,0]
        plt.plot(self.time, r)
        plt.xlabel(r"Time [$t/\tau$]", **label_size)
        plt.ylabel(r"Position [r/$\sigma$]", **label_size)
        plt.show()
        
class PlotVelocities(Tasks):
    """ Plot position
    """
    def __init__(self, solver):
        self.solver = solver
        self.time = solver.time
        self.v = np.zeros((solver.N, solver.numparticles, solver.numdimensions))
        
    def __str__(self):
        return "Plotting velocities as a function of time."
    
    def _update(self, state, step):
        """ Append to position array.
        """
        self.v[step] = state.v
        
    def __call__(self):
        """ Plot position as a function of time.
        """
        v = self.v[:,0,0]
        plt.plot(self.time, v)
        plt.xlabel(r"Time [$t/\tau$]", **label_size)
        plt.ylabel("Velocity", **label_size)
        plt.show()
        
class PlotEnergy(Tasks):
    """ Plot energy
    """
    def __init__(self, solver):
        self.solver = solver
        self.time = solver.time
        self.k = np.zeros(solver.N)
        self.u = np.zeros(solver.N)
        
    def __str__(self):
        return "Plotting energies as a function of time."
       
    @staticmethod
    def potentialEnergy(u):
        """ Calculates the total potential energy, based on 
        the potential energies of all particles stored in the matrix
        u. Shifts the potential according to the cutoff.
        
        Parameters
        ----------
        u : ndarray
            array containing the potential energy of all the particles.
        cutoff : float
            cutoff distance: maximum length of the interactions. 3 by default.
                          
        Returns
        -------
        float
            total potential energy
        """
        u[u == np.inf] = 0
        return 4 * (np.sum(u)) # - potstate.cutoffPowTwelveInv + potstate.cutoffPowSixInv)
        
    @staticmethod
    def kineticEnergy(v):
        """ Returns the total kinetic energy for each timestep.
        This function is never called in the integration loop, but can 
        be used to obtain the energy of the system afterwards.
        
        Parameters
        ----------
        v : ndarray
            velocity array
        
        Returns
        -------
        1darray
            total kinetic energy at all timesteps
        """
        return (v**2).sum()/2
       
    #@classmethod
    def _update(self, state, step):
        """ Update kinetic and potential energy every timestep
        """
        self.u[step] = self.potentialEnergy(state.u)
        self.k[step] = self.kineticEnergy(state.v)
       
    def __call__(self):
        """ This function plots the kinetic, potential and total energy.
        The kinetic energy is taken from the kineticEnergy function,
        while the potential energy is taken from the specified potential
        (which in our case is Lennard-Jones).
        """
        
        e = self.u + self.k                  # Total energy
        plt.plot(self.time, self.k, label="Kinetic")
        plt.plot(self.time, self.u, label="Potential")
        plt.plot(self.time, e, label="Total energy")
        plt.legend(loc="best", fontsize=14)
        plt.xlabel(r"Time [$t/\tau$]", **label_size)
        plt.ylabel(r"Energy [$\varepsilon$]", **label_size)
        plt.show()
       
class PlotDistance(Tasks):
    """ Plot distance between all the particles
    """
    def __init__(self, solver):
        self.solver = solver
        self.time = solver.time
        self.numparticles=solver.numparticles
        
        assert self.numparticles <= 4, (
                "Cannot plot the distance for more than 4 particles.")
        
        self.d = np.zeros((solver.N, self.numparticles, self.numparticles))
        
    def __str__(self):
        return "Plotting distances as a function of time."
       
    #@classmethod
    def _update(self, state, step):
        self.d[step] = np.sqrt(state.d)
        
    def __call__(self):
        """ Plot distance between all particles. The plot will contain a 
        graph for each particle pair, giving N(N-1)/2 graphs. It is 
        recommended to use just for a small number of particles.
        """
        for i in range(self.numparticles):
            for j in range(i):
                plt.plot(self.time, self.d[:,i,j], label="$i={}$, $j={}$".format(i,j))
        plt.legend(loc="best", fontsize=14)
        plt.xlabel(r"Time [$t/\tau$]", **label_size)
        plt.ylabel("$r_{ij} [r/\sigma]$", **label_size)
        plt.show()
        
class MSD(Tasks):
    """ Mean square distance
    """
    def __init__(self, solver):
        self.solver = solver
        self.time = solver.time
        self.numparticles = solver.numparticles
        self.msd = np.zeros(solver.N)
        self.r0 = None
        
    #@classmethod
    def _update(self, state, step):
        if self.r0 is None:
            self.r0 = state.r
        dis = state.r + state.c - self.r0
        self.msd[step] = (dis**2).sum()/self.numparticles
        
    def __call__(self):
        """ Plot the mean square displacement as a function of time. It is
        calculated using the formula
        
        <r^2(t)> = <(r(t)-r(t0))^2>
        """
        plt.plot(self.time, self.msd)
        plt.xlabel(r"Time [$t/\tau$]", **label_size)
        plt.ylabel("Mean square displacement", **label_size)
        plt.show()
        
class AutoCorrelation(Tasks):
    """ Velocity auto-correlation function.
    """
    def __init__(self, solver):
        self.solver = solver
        self.N = solver.N
        self.time = solver.time
        self.numparticles = solver.numparticles
        self.A = np.zeros(self.N)
        self.v0 = None
        
    #@classmethod
    def _update(self, state, step):
        if self.v0 is None:
            self.v0 = state.v
            self.v02 = np.einsum('ij,ij',self.v0,self.v0) * self.N
            
        self.A[step] = np.einsum('ij,ij',state.v,self.v0) / self.v02
        
    def __call__(self):
        """ Plot the mean square displacement as a function of time. It is
        calculated using the formula
        
        <A(t)> = <(v(t)*v(t0))/(v(t0)*v(t0))>
        """
        plt.plot(self.time, self.A)
        plt.xlabel(r"Time [$t/\tau$]", **label_size)
        plt.ylabel("Velocity Auto-correlation", **label_size)
        plt.show()
        
class RDF(Tasks):
    """ Radial distribution function
    
    Parameters
    ----------
    bin_edges : ndarray
        edges of bins. Typically np.linspace(0, rc, num_bins+1)
        for some cut-off rc.
    """
    def __init__(self, solver, num_bins=100):
        self.solver = solver
        self.N = solver.N
        self.num_bins = num_bins
        self.numparticles = solver.numparticles
        self.bin_edges = np.linspace(0, 3, num_bins+1)
        self.rdf = np.zeros(self.num_bins)
        self.V = 1000
    
    def _update(self, state, step):
        if step == self.N-1:
            r = state.r
            bin_centres = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])
            bin_sizes = self.bin_edges[1:] - self.bin_edges[:-1]
            n = np.zeros_like(bin_sizes)
            for i in range(self.numparticles):
                dr = np.linalg.norm(r - r[i], axis=1)    # Distances from atom i.
                n += np.histogram(dr, bins=self.bin_edges)[0] # Count atoms within each
                                                         # distance interval.
            # Equation (7) on the preceding page:
            rdf = self.V / self.numparticles**2 * n / (4 * np.pi * bin_centres**2 * bin_sizes)
            return rdf
            
    def __call__(self):
        plt.plot(self.bin_edges[:-1], self.rdf)
        plt.xlabel(r"$r$ [$r/\sigma$]", **label_size)
        plt.ylabel(r"$g(r)$", **label_size)
        plt.show()
    
