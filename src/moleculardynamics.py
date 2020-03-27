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
    positions : array_like, list
        nested list with all coordinates of all
        particles. Face-centered cube is default.
    velocity : array_like, list
        nested list with all velocities of all
        particles. No velocity is default.
    cells : int
        number of unit cells
    lencell : float
        length of unit cell
    numdimensions : int
        number of dimensions
    T : float
        total time
    dt : float
        time step
    size : int
        label size
    
    cells, lencell and numdimensions are only needed by fcc
    """
    def __init__(self, positions='fcc', 
                       velocity=None, 
                       cells=2, 
                       lenbulk=20, 
                       numdimensions=3, 
                       T=5, 
                       dt=0.01, 
                       size=16):
        
        # Define time scale and number of steps
        self.T = T
        self.dt = dt
        self.N = int(T/dt)
        self.time = np.linspace(0, T, self.N+1)
        
        # Initialize positions
        if positions=='fcc':
            self.face_centered_cube(cells, lenbulk, numdimensions)
        elif type(positions)==list:
            self.numparticles = len(positions)
            self.numdimensions = len(positions[0])
            self.r = np.zeros((self.N+1, self.numparticles, self.numdimensions))
            self.r[0] = positions
        elif positions is None:
            raise TypeError("Initial positions are not defined")
        else:
            raise TypeError("Initial positions needs to be a list of positions")
        
        self.dumpPositions(0, "../data/initialPositions.data")
        
        # Initialize velocities
        if velocity==None:
            self.v = np.zeros((self.N+1, self.numparticles, self.numdimensions))
        elif velocity=="gauss":
            self.v = np.random.normal(0, 1, size=(self.N+1, self.numparticles, self.numdimensions))
        elif type(velocity) == list:
            self.v[0] = velocity
        
        # print to terminal
        self.print_to_terminal()
        
        # for plotting
        self.size = size                        # Label size in plots
        self.label_size = {"size":str(size)}    # Dictionary with size
        plt.style.use("bmh")                    # Beautiful plots
        plt.rcParams["font.family"] = "Serif"   # Font
        
    def print_to_terminal(self):
        """ Print information to terminal
        """
        print("\n\n" + 10 * "=", " SYSTEM INFORMATION ", 10 * "=")
        print("Number of particles:  ", self.numparticles)
        print("Number of dimensions: ", self.numdimensions)
        print("Total time:           ", self.T, " ps")
        print("Timestep:             ", self.dt, " ps")
        print(42 * "=" + "\n\n")
        
        
    def face_centered_cube(self, n, L, dim):
        """ Creating a face-centered cube of n^dim unit cells with
        4 particles in each unit cell. The number of particles
        then becomes (dim+1) * n ^ dim. Each unit cell has a 
        length d. L=nd
        
        Parameters
        ----------
        n : int
            number of unit cells in each dimension
        L : float
            length of bulk
        dim : int
            number of dimensions
            
        Returns
        -------
        2darray
            initial particle configuration
        """
        self.numparticles = (dim+1) * n ** dim
        self.numdimensions = dim
        self.r = np.zeros((self.N+1, self.numparticles, dim))
        counter = 0
        if dim==1:
            for i in range(n):
                self.r[0,counter+0] = [i]
                self.r[0,counter+1] = [0.5+i]
                counter +=2
        elif dim==2:
            for i in range(n):
                for j in range(n):
                    self.r[0,counter+0] = [i, j]
                    self.r[0,counter+1] = [i, 0.5+j]
                    self.r[0,counter+2] = [0.5+i, j]
                    counter += 3
        elif dim==3:
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        self.r[0,counter+0] = [i, j, k]
                        self.r[0,counter+1] = [i, 0.5+j, 0.5+k]
                        self.r[0,counter+2] = [0.5+i, j, 0.5+k]
                        self.r[0,counter+3] = [0.5+i, 0.5+j, k]
                        counter += 4
        else:
            raise ValueError("The number of dimensions needs to be in [1,3]")
        self.r[0] *= L/n
        return self.r[0]
            
    @staticmethod
    def calculateDistanceMatrix(r):
        """ Compute the distance matrix (squared) at timestep t. In the
        integration loop, we only need the distance squared, which 
        means that we do not need to take the square-root of the 
        distance. This save some cpu time.
        
        Parameters
        ----------
        r : ndarray
            spatial coordinates at some timestep
            
        Returns
        -------
        dr : ndarray
            distance vector between all the particles
        distanceSqrd : ndarray
            distance between all particles squared
        distancePowSixInv : ndarray
            distance between all particles to the power of six inverse
        distancePowTwelveInv : ndarray
            distance between all particles to the power of twelve inverse
        """
        x, y = r[:,np.newaxis,:], r[np.newaxis,:,:]
        dr = x - y
        distanceSqrd = np.einsum('ijk,ijk->ij',dr,dr)              # r^2
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))      # 1/r^6
        distancePowTwelveInv = distancePowSixInv**2                # 1/r^12
        return dr, distanceSqrd, distancePowSixInv, distancePowTwelveInv
        
    def lennardJones(self, t):
        """ Lennard-Jones inter-atomic force. This is used in the
        integration loop to calculate the acceleration of particles. 
        
        Parameters
        ----------
        t : int
            current time step.
            
        Returns
        -------
        ndarray
            The netto force acting on every particle
        """
        dr, d, l, m = self.calculateDistanceMatrix(self.r[t])
        if self.distance: 
            self.d[t] = d
        if self.poteng: 
            self.u[t] = self.lennardJonesEnergy(m - l)
        factor = np.divide(2 * m - l, d)            # (2/r^12 - 1/r^6)/r^2
        factor[factor == np.inf] = 0
        return - 24 * np.einsum('ij,ijk->jk',factor,dr)
        
    @staticmethod
    def lennardJonesEnergy(u):
        """ Calculates the potential energy at timestep t, based on 
        the potential energies of all particles stored in the matrix
        u.
        
        Parameters
        u : ndarray
            array containing the potential energy of all the particles.
        """
        u[u == np.inf] = 0
        return 2 * np.sum(u)       # Multiply with 4 / 2
        
    def kineticEnergy(self):
        """ Returns the total kinetic energy for each timestep.
        This function is never called in the integration loop, but can 
        be used to obtain the energy of the system afterwards.
        
        Returns
        -------
        1darray
            total kinetic energy at all timesteps
        """
        return (self.v**2).sum(axis=1).sum(axis=1)/2
        
    def forwardEuler(self, t, potential):
        """ Forward-Euler numerical integration. This function gets the
        acceleration from a potential function. In our case, this
        potential is Lennard-Jones. Based on the acceleration, it
        finds the velocity at the current timestep t using the 
        Forward-Euler integration scheme. 
        
        Parameters
        ----------
        t : int
            current timestep
        potential : def
            inter-atomic potential (Lennard-Jones)
        """
        # calculate force acting on all the particles
        a = potential(t)
        self.v[t+1] = self.v[t] + a * self.dt
        self.r[t+1] = self.r[t] + self.v[t] * self.dt
        
    def eulerChromer(self, t, potential):
        """ Euler-Chromer numerical integration. This function gets the
        acceleration from a potential function. In our case, this
        potential is Lennard-Jones. Based on the acceleration, it
        finds the velocity at the current timestep t using the 
        Euler-Chromer integration scheme. 
        
        Parameters
        ----------
        t : int
            current timestep
        potential : def
            inter-atomic potential (Lennard-Jones)
        """
        a = potential(t)
        self.v[t+1] = self.v[t] + a * self.dt
        self.r[t+1] = self.r[t] + self.v[t+1] * self.dt
        
    def velocityVerlet(self, t, potential):
        """ Velocity-Verlet numerical integration. This function gets the
        acceleration from a potential function. In our case, this
        potential is Lennard-Jones. Based on the acceleration, it
        finds the velocity at the current timestep t using the 
        Velocity-Verlet integration scheme. 
        
        Parameters
        ----------
        t : int
            current timestep
        potential : def
            inter-atomic potential (Lennard-Jones)
        """
        a = potential(t)
        self.r[t+1] = self.r[t] + self.v[t] * self.dt + 0.5 * a * self.dt**2
        a_new = potential(t+1)
        self.v[t+1] = self.v[t] + 0.5 * (a_new + a) * self.dt
        
    def dumpPositions(self, t, dumpfile):
        """ Dumping positions at timestep t to a dumpfile. We use the xyz-
        format, which can easily be visualized using Ovito.
        
        Parameters
        ----------
        t : int
            current timestep
        dumpfile : str
            name and address of dumpfile
        """
        dat = np.column_stack((self.numparticles * ['Ar'], self.r[t]))
        np.savetxt(dumpfile, dat, header="{}\ntype x y z".format(self.numparticles), fmt="%s", comments='')
    
    def simulate(self, potential, integrator, poteng=False, distance=False, cutoff=3.0, dumpfile=None):
        """ Integration loop. Computes the time-development of position and 
        velocity using a given integrator and inter-atomic potential.
        
        Parameters
        ----------
        potential : def
            function defining the inter-atomic potential
        integrator : def
            function defining the integrator
        poteng : bool or int
            boolean saying whether or not the potential
            energy should be calculated and stored.
        distance : bool or int
            boolean saying whether or not the distance matrix should be stored. 
        cutoff : float
            cutoff distance
        dumpfile : str
            filename that all the positions should be dumped to. If not 
            specified, positions are not dumped.  
        """
        self.cutoff_sqrd = cutoff * cutoff      # Use cutoff^2 and d^2 only
        self.poteng = poteng
        self.distance = distance
        if distance: 
            self.d = np.zeros((self.N+1, self.numparticles, self.numparticles))
        if poteng: 
            self.u = np.zeros(self.N+1) # Potential energy
        if dumpfile is not None: 
            f=open(dumpfile,'ab')       # Open dumpfile
            self.dumpPositions(0,f)     # Dump initial positions
        from tqdm import tqdm
        for t in tqdm(range(self.N)):   # Integration loop
            integrator(t, potential)    # integrate to find velocities and positions
            if dumpfile is not None: 
                self.dumpPositions(t+1,f) # dump positions to file
        if dumpfile is not None: 
            f.close()      # Close dumpfile
        if distance: 
            self.d[self.N] = self.calculateDistanceMatrix(self.r[-1])[1]    # Calculate final distance 
        if poteng:
            dr, d, l, m = self.calculateDistanceMatrix(self.r[-1])
            self.u[self.N] = self.lennardJonesEnergy(m - l)
        
    def plot_distance(self):
        """ Plot distance between all particles. The plot will contain a 
        graph for each particle pair, giving N(N-1)/2 graphs. It is 
        recommended to use just for a small number of particles.
        """
        distance = np.sqrt(self.d)
        for i in range(self.numparticles):
            for j in range(i):
                plt.plot(self.time, distance[:,i,j], label="$i={}$, $j={}$".format(i,j))
        plt.legend(loc="best", fontsize=self.size)
        plt.xlabel(r"Time [$t'/\tau$]", **self.label_size)
        plt.ylabel("$r_{ij}$", **self.label_size)
        plt.show()
        
    def plot_energy(self):
        """ This function plots the kinetic, potential and total energy.
        The kinetic energy is taken from the kineticEnergy function,
        while the potential energy is taken from the specified potential
        (which in our case is Lennard-Jones).
        """
        k = self.kineticEnergy()        # Kinetic energy
        e = k + self.u                  # Total energy
        plt.plot(self.time, k, label="Kinetic")
        plt.plot(self.time, self.u, label="Potential")
        plt.plot(self.time, e, label="Total energy")
        plt.legend(loc="best", fontsize=self.size)
        plt.xlabel(r"Time [$t'/\tau$]", **self.label_size)
        plt.ylabel(r"Energy [$\varepsilon$]", **self.label_size)
        plt.show()

if __name__ == "__main__":
    # EXAMPLE: TWO PARTICLES IN ONE DIMENSION
    obj = MDSolver(positions=[[0.0], [1.5]], T=5, dt=0.01)
    obj.simulate(potential=obj.lennardJones, 
                 integrator=obj.eulerChromer,
                 distance=True,
                 poteng=True)
    obj.plot_distance()
    obj.plot_energy()
