import numpy as np
import matplotlib.pyplot as plt

class Tasks:
    """ Which physical observables to display
    """
    def __init__(self, solver):
        self.solver = solver
        
        # for plotting
        self.label_size = {"size":14}    # Dictionary with size
        plt.style.use("bmh")                    # Beautiful plots
        plt.rcParams["font.family"] = "Serif"   # Font
        
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
        self.f = open(dumpfile,'w')       # Open dumpfile
        
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

        
class PlotEnergy(Tasks):
    """ Plot energy
    """
    def __init__(self, solver):
        self.solver = solver
        self.k = np.zeros(solver.N)
        self.u = np.zeros(solver.N)
       
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
        plt.plot(self.solver.time, self.k, label="Kinetic")
        plt.plot(self.solver.time, self.u, label="Potential")
        plt.plot(self.solver.time, e, label="Total energy")
        plt.legend(loc="best", fontsize=14)
        plt.xlabel(r"Time [$t'/\tau$]")#, **self.label_size)
        plt.ylabel(r"Energy [$\varepsilon$]")#, **self.label_size)
        plt.show()
       
class PlotDistance(Tasks):
    """ Plot distance between all the particles
    """
    def __init__(self, solver):
        self.solver = solver
        self.numparticles=solver.numparticles
        self.d = np.zeros((solver.N, self.numparticles, self.numparticles))
       
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
                plt.plot(self.solver.time, self.d[:,i,j], label="$i={}$, $j={}$".format(i,j))
        plt.legend(loc="best", fontsize=14)
        plt.xlabel(r"Time [$t'/\tau$]")#, **self.label_size)
        plt.ylabel("$r_{ij}$")#, **self.label_size)
        plt.show()
        
class MSD(Tasks):
    """ Mean square distance
    """
    def __init__(self, solver):
        self.solver=solver
        self.msd=np.zeros(solver.N)
        self.r0=None
        
    #@classmethod
    def _update(self, state, step):
        if self.r0 is None:
            self.r0 = state.r
        dis = state.r + state.c - self.r0
        self.msd[step] = (dis**2).sum()/self.solver.numparticles
        
    def __call__(self):
        """ Plot the mean square displacement as a function of time. It is
        calculated using the formula
        
        <r^2(t)> = <(r(t)-r(t0))^2>
        """
        plt.plot(self.solver.time, self.msd)
        plt.xlabel(r"Time [$t'/\tau$]")#, **self.label_size)
        plt.ylabel("Mean square displacement")#, **self.label_size)
        plt.show()
        
class 
