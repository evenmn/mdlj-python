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
    boundaries : str
        string specifying the boundary conditions to be used in each 
        direction. o - open, r - reflective, p - periodic
    cells : int
        number of unit cells
    lenbox : float
        length of simulation box. Applies in all dimensions
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
                       boundaries='ooo',
                       cells=2, 
                       lenbox=20, 
                       numdimensions=3, 
                       T=5, 
                       dt=0.01, 
                       size=14):
        
        self.lenbox = lenbox
        self.boundaries = boundaries
        
        # Define time scale and number of steps
        self.T = T
        self.dt = dt
        self.N = int(T/dt)
        self.time = np.linspace(0, T, self.N+1)
        
        # Initialize positions
        if positions=='fcc':
            self.face_centered_cube(cells, lenbox, numdimensions)
        elif type(positions) == list:
            self.numparticles = len(positions)
            self.numdimensions = len(positions[0])
            self.r = np.zeros((self.N+1, self.numparticles, self.numdimensions))
            self.r[0] = positions
        else:
            raise TypeError("Initial positions needs to be a list of positions")
        
        self.dumpPositions(0, "../data/initialPositions.data")
        
        # Initialize velocities
        if velocity==None:
            self.v = np.zeros((self.N+1, self.numparticles, self.numdimensions))
        elif velocity=="gauss":
            self.v = np.random.normal(0, 1, size=(self.N+1, self.numparticles, self.numdimensions))
        elif type(velocity) == list:
            self.v = np.zeros((self.N+1, self.numparticles, self.numdimensions))
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
        print("Total time:           ", self.T, "\tps")
        print("Timestep:             ", self.dt, "\tps")
        print(42 * "=" + "\n\n")
        
        
    def face_centered_cube(self, cells, lenbox, dim):
        """ Creating a face-centered cube of n^dim unit cells with
        4 particles in each unit cell. The number of particles
        then becomes (dim+1) * n ^ dim. Each unit cell has a 
        length d. L=nd
        
        Parameters
        ----------
        cells : int
            number of unit cells in each dimension
        lenbox : float
            length of box
        dim : int
            number of dimensions
            
        Returns
        -------
        2darray
            initial particle configuration
        """
        self.numparticles = (dim+1) * cells ** dim
        self.numdimensions = dim
        self.r = np.zeros((self.N+1, self.numparticles, dim))
        counter = 0
        if dim==1:
            for i in range(cells):
                self.r[0,counter+0] = [i]
                self.r[0,counter+1] = [0.5+i]
                counter +=2
        elif dim==2:
            for i in range(cells):
                for j in range(cells):
                    self.r[0,counter+0] = [i, j]
                    self.r[0,counter+1] = [i, 0.5+j]
                    self.r[0,counter+2] = [0.5+i, j]
                    counter += 3
        elif dim==3:
            for i in range(cells):
                for j in range(cells):
                    for k in range(cells):
                        self.r[0,counter+0] = [i, j, k]
                        self.r[0,counter+1] = [i, 0.5+j, 0.5+k]
                        self.r[0,counter+2] = [0.5+i, j, 0.5+k]
                        self.r[0,counter+3] = [0.5+i, 0.5+j, k]
                        counter += 4
        else:
            raise ValueError("The number of dimensions needs to be in [1,3]")
        # Scale initial positions correctly
        self.r[0] *= lenbox/cells
        return self.r[0]
        
        
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
    
    def simulate(self, potential, integrator, poteng=True, distance=False, dumpfile=None):
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
        dumpfile : str
            filename that all the positions should be dumped to. If not 
            specified, positions are not dumped.
        """
        self.potential = potential
        
        a, u, d = potential(self.r[0])
        if distance: 
            self.d = np.zeros((self.N, self.numparticles, self.numparticles))
            self.d[0] = d
        if poteng: 
            self.u = np.zeros(self.N) # Potential energy
            self.u[0] = u
        if dumpfile is not None: 
            f = open(dumpfile,'w')       # Open dumpfile
            self.dumpPositions(0,f)     # Dump initial positions
        from tqdm import tqdm
        for t in tqdm(range(self.N)):   # Integration loop
            # integrate to find velocities and positions
            self.r[t+1], self.v[t+1], a, u, d = integrator(self.r[t], self.v[t], a)
            #self.r[t+1], self.v[t+1] = self.boundary(r, v)
            if dumpfile is not None: 
                self.dumpPositions(t+1,f) # dump positions to file
            if distance:
                self.d[t] = d
            if poteng:
                self.u[t] = u
        if dumpfile is not None: 
            f.close()      # Close dumpfile
        
    def plot_distance(self):
        """ Plot distance between all particles. The plot will contain a 
        graph for each particle pair, giving N(N-1)/2 graphs. It is 
        recommended to use just for a small number of particles.
        """
        distance = np.sqrt(self.d)
        for i in range(self.numparticles):
            for j in range(i):
                plt.plot(self.time[:-1], distance[:,i,j], label="$i={}$, $j={}$".format(i,j))
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
        time = self.time[:-1]
        k = self.kineticEnergy()[:-1]   # Kinetic energy
        e = k + self.u                  # Total energy
        plt.plot(time, k, label="Kinetic")
        plt.plot(time, self.u, label="Potential")
        plt.plot(time, e, label="Total energy")
        plt.legend(loc="best", fontsize=self.size)
        plt.xlabel(r"Time [$t'/\tau$]", **self.label_size)
        plt.ylabel(r"Energy [$\varepsilon$]", **self.label_size)
        plt.show()
            

if __name__ == "__main__":
    # EXAMPLE: TWO PARTICLES IN ONE DIMENSION INITIALLY SEPARATED BY 1.5 SIGMA
    from potential import LennardJones
    from integrator import VelocityVerlet, EulerChromer

    solver = MDSolver(positions=[[0.0], [1.5]], T=5, dt=0.01)
    solver.simulate(potential=LennardJones(cutoff=3), 
                    integrator=EulerChromer(solver),
                    distance=True,
                    dumpfile="../data/2N_1D_1.5S.data")
    solver.plot_distance()
    solver.plot_energy()
