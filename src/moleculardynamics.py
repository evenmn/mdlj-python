import numpy as np
import matplotlib.pyplot as plt
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
    
    from initpositions import FCC
    from initvelocities import Zero
    from boundaryconditions import Open
    
    def __init__(self, positions=FCC(cells=1, lenbulk=3), 
                       velocities=Zero(), 
                       boundaries=Open(),
                       T=5, 
                       dt=0.01):
        
        self.boundaries = boundaries
        
        # Define time scale and number of steps
        self.T = T
        self.dt = dt
        self.N = int(T/dt)
        self.time = np.linspace(0, T, self.N)
        
        # Initialize positions
        r0 = positions()
        self.numparticles = len(r0)
        self.numdimensions = len(r0[0])
        self.r = np.zeros((self.N+1, self.numparticles, self.numdimensions))
        self.r[0] = r0
        self.dumpPositions(r0, "../data/initialPositions.data")
        
        # Initialize velocities
        self.v = np.zeros(self.r.shape)
        self.v[0] = velocities(self.numparticles, self.numdimensions)
        
        # print to terminal
        self.print_to_terminal()
        
        # for plotting
        self.label_size = {"size":14}    # Dictionary with size
        plt.style.use("bmh")                    # Beautiful plots
        plt.rcParams["font.family"] = "Serif"   # Font
        
    def print_to_terminal(self):
        """ Print information to terminal
        """
        print("\n\n" + 14 * "=", " SYSTEM INFORMATION ", 14 * "=")
        print("Number of particles:  ", self.numparticles)
        print("Number of dimensions: ", self.numdimensions)
        print("Boundary conditions:  ", self.boundaries)
        print("Total time:           ", self.T, "\tps")
        print("Timestep:             ", self.dt, "\tps")
        print(50 * "=" + "\n\n")
        
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
        return (v**2).sum(axis=1).sum(axis=1)/2
        
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
               
    @staticmethod    
    def print_simulation(potential, integrator, poteng, distance, dumpfile):
        """ Print information to terminal when starting a simulation
        
        Parameters
        ----------
        potential : obj
            object defining the inter-atomic potential
        integrator : obj
            object defining the integrator
        poteng : bool or int
            boolean saying whether or not the potential
            energy should be calculated and stored.
        distance : bool or int
            boolean saying whether or not the distance matrix should be stored. 
        dumpfile : str
            filename that all the positions should be dumped to. If not 
            specified, positions are not dumped.
        """
        print("\n\n" + 12 * "=", " SIMULATION INFORMATION ", 12 * "=")
        print("Potential:            ", potential)
        print("Integrator:           ", integrator)
        print("Potential energy:     ", poteng)
        print("Store distance:       ", distance)
        print("Dump file:            ", dumpfile)
        print(50 * "=" + "\n\n")
    
    def __call__(self, potential, 
                       integrator, 
                       poteng=True, 
                       distance=False, 
                       dumpfile=None):
        """ Integration loop. Computes the time-development of position and 
        velocity using a given integrator and inter-atomic potential.
        
        Parameters
        ----------
        potential : obj
            object defining the inter-atomic potential
        integrator : obj
            object defining the integrator
        poteng : bool or int
            boolean saying whether or not the potential
            energy should be calculated and stored.
        distance : bool or int
            boolean saying whether or not the distance matrix should be stored. 
        dumpfile : str
            filename that all the positions should be dumped to. If not 
            specified, positions are not dumped.
        """
        self.potential = potential
        
        # Print information
        self.print_simulation(potential, integrator, poteng, distance, dumpfile)
        
        # Compute initial acceleration, potential energy and distance matrix
        a, u, d = potential(self.r[0])
        
        # Dump positions to dumpfile if dumpfile is defined
        if dumpfile is not None: 
            f = open(dumpfile,'w')       # Open dumpfile
            self.dumpPositions(self.r[0],f)     # Dump initial positions
        
        # Store distance matrix if distance=True
        if distance: 
            self.d = np.zeros((self.N, self.numparticles, self.numparticles))
            self.d[0] = d
            
        # Store potential energy if poteng=True
        if poteng: 
            self.u = np.zeros(self.N) # Potential energy
            self.u[0] = u
            
        # Integration loop
        from tqdm import tqdm
        for t in tqdm(range(self.N)):   # Integration loop
            self.r[t+1], self.v[t+1], a, u, d = integrator(self.r[t], self.v[t], a)
            
            # Dump positions to dumpfile if dumpfile is defined
            if dumpfile is not None: 
                self.dumpPositions(self.r[t+1],f) # dump positions to file
                
            # Store distance matrix if distance=True
            if distance:
                self.d[t] = d
                
            # Store potential energy if poteng=True
            if poteng:
                self.u[t] = u
                
        # Close dumpfile
        if dumpfile is not None: 
            f.close()
        
    def plot_distance(self):
        """ Plot distance between all particles. The plot will contain a 
        graph for each particle pair, giving N(N-1)/2 graphs. It is 
        recommended to use just for a small number of particles.
        """
        distance = np.sqrt(self.d)
        for i in range(self.numparticles):
            for j in range(i):
                plt.plot(self.time, distance[:,i,j], label="$i={}$, $j={}$".format(i,j))
        plt.legend(loc="best", fontsize=14)
        plt.xlabel(r"Time [$t'/\tau$]", **self.label_size)
        plt.ylabel("$r_{ij}$", **self.label_size)
        plt.show()
        
    def plot_energy(self):
        """ This function plots the kinetic, potential and total energy.
        The kinetic energy is taken from the kineticEnergy function,
        while the potential energy is taken from the specified potential
        (which in our case is Lennard-Jones).
        """
        k = self.kineticEnergy(self.v)[:-1]   # Kinetic energy
        e = k + self.u                  # Total energy
        plt.plot(self.time, k, label="Kinetic")
        plt.plot(self.time, self.u, label="Potential")
        plt.plot(self.time, e, label="Total energy")
        plt.legend(loc="best", fontsize=14)
        plt.xlabel(r"Time [$t'/\tau$]", **self.label_size)
        plt.ylabel(r"Energy [$\varepsilon$]", **self.label_size)
        plt.show()
        
    def plot_temperature(self):
        """ Plot the temperature as a function of time. The temperature
        is calculated using the formula T=v^2/ND.
        """
        k = self.kineticEnergy(self.v)[:-1]
        T = k * 2 * 119.7 / (self.numparticles * self.numdimensions)
        plt.plot(self.time, T)
        plt.xlabel(r"Time [$t'/\tau$]", **self.label_size)
        plt.ylabel(r"Temperature [K]", **self.label_size)
        plt.show()

if __name__ == "__main__":
    # EXAMPLE: TWO PARTICLES IN ONE DIMENSION INITIALLY SEPARATED BY 1.5 SIGMA
    from potential import LennardJones
    from integrator import EulerChromer
    from initpositions import SetPositions

    solver = MDSolver(positions=SetPositions([[0.0], [1.5]]), 
                      T=5, 
                      dt=0.01)
    solver(potential=LennardJones(solver, cutoff=2.5), 
           integrator=EulerChromer(solver),
           distance=True,
           dumpfile="../data/2N_1D_1.5S.data")
    solver.plot_distance()
    solver.plot_energy()
