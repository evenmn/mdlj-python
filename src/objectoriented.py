import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class MolecularDynamics:
    def __init__(self, positions='fcc', 
                       velocity=None, 
                       cells=2, 
                       lencell=3, 
                       numdimensions=3, 
                       T=5, 
                       dt=0.01, 
                       size=16):
        '''
        Initialize the MolecularDynamics class. This includes defining the
        time scales, initialize positions and velocities, and define
        matplotlib fixes.
        
        Arguments:
        ----------
        positions       {list}  :   Nested list with all coordinates of all
                                    particles. Face-centered cube is default.
        velocity        {list}  :   Nested list with all velocities of all
                                    particles. No velocity is default.
        cells           {int}   :   Number of unit cells
        lencell         {float} :   Length of unit cell
        numdimensions   {int}   :   Number of dimensions
        T               {float} :   Total time
        dt              {float} :   Time step
        size            {int}   :   Label size
        
        cells, lencell and numdimensions are only needed by fcc
        '''
        
        # Define time scale and number of steps
        self.T = T
        self.dt = dt
        self.N = int(T/dt)
        self.time = np.linspace(0, T, self.N+1)
        
        # Initialize positions
        if positions=='fcc':
            self.face_centered_cube(cells, lencell, numdimensions)
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

        # declare distance matrix
        self.d = np.zeros((self.N+1, self.numparticles, self.numparticles))
        
        # print to terminal
        self.print_to_terminal()
        
        # for plotting
        self.size = size                        # Label size in plots
        self.label_size = {"size":str(size)}    # Dictionary with size
        plt.style.use("bmh")                    # Beautiful plots
        plt.rcParams["font.family"] = "Serif"   # Font
        
    def print_to_terminal(self):
        '''
        Print information to terminal
        '''
        print("\n\n" + 10 * "=", " SYSTEM INFORMATION ", 10 * "=")
        print("Number of particles:  ", self.numparticles)
        print("Number of dimensions: ", self.numdimensions)
        print("Total time:           ", self.T, " ps")
        print("Timestep:             ", self.dt, " ps")
        print(42 * "=" + "\n\n")
        
        
    def face_centered_cube(self, n, d, dim):
        '''
        Creating a face-centered cube of n^dim unit cells with
        4 particles in each unit cell. The number of particles
        then becomes (dim+1) * n ^ dim. Each unit cell has a 
        length d.
        
        Arguments:
        ----------
        n               {int}   :   Number of unit cells in each dimension
        d               {float} :   Length of a unit cell
        dim             {int}   :   Number of dimensions
        '''
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
        self.r[0] *= d
        return self.r[0]
            
    def calculateDistanceMatrix(self, t):
        ''' 
        Compute the distance matrix (squared) at timestep t. In the
        integration loop, we only need the distance squared, which 
        means that we do not need to take the square-root of the 
        distance. This save some cpu time.
        
        Arguments:
        ----------
        t               {int}   :   Current time step.
        '''
        x = self.r[t][:,np.newaxis,:]
        y = self.r[t][np.newaxis,:,:]
        dr = x - y
        del x, y
        self.d[t] = np.einsum('ijk,ijk->ij',dr,dr)
        return dr
        
    def lennardJones(self, t):
        ''' 
        Lennard-Jones inter-atomic force. This is used in the
        integration loop to calculate the acceleration of particles. 
        
        Arguments:
        ----------
        t               {int}   :   Current time step.
        '''
        dr = self.calculateDistanceMatrix(t)
        l = np.nan_to_num(self.d[t]**(-3))          # 1/r^6
        m = l**2                                    # 1/r^12
        factor = np.divide(2 * m - l, self.d[t])    # (2/r^12 - 1/r^6)/r^2
        del l, m
        factor[factor == np.inf] = 0
        return - 24 * np.einsum('ij,ijk->jk',factor,dr)
        del dr
        
    def lennardJonesEnergy(self):
        ''' 
        Returns the potential energy from the Lennard-Jones potential. 
        This function is never called in the integration loop, but can 
        be used to obtain the energy of the system afterwards.
        '''
        l = np.nan_to_num(self.d**(-3))             # 1/r^6
        m = l**2                                    # 1/r^12
        u = m - l
        del l, m
        u[u == np.inf] = 0
        p = 4 * u.sum(axis=1).sum(axis=1)/2
        del u
        return p - np.min(p)
        
    def kineticEnergy(self):
        ''' 
        Returns the total kinetic energy for each timestep.
        This function is never called in the integration loop, but can 
        be used to obtain the energy of the system afterwards.
        '''
        kPar = np.sum(np.square(self.v), axis=2)
        return np.sum(kPar, axis=1)/2
        del kPar
        
    def forwardEuler(self, t, potential):
        ''' 
        Forward-Euler numerical integration. This function gets the
        acceleration from a potential function. In our case, this
        potential is Lennard-Jones. Based on the acceleration, it
        finds the velocity at the current timestep t using the 
        Forward-Euler integration scheme. 
        
        Arguments:
        ----------
        t               {int}   :   Current timestep.
        potential       {func}  :   Inter-atomic potential (Lennard-Jones)
        '''
        # calculate force acting on all the particles
        a = potential(t)
        self.v[t+1] = self.v[t] + a * self.dt
        self.r[t+1] = self.r[t] + self.v[t] * self.dt
        
    def eulerChromer(self, t, potential):
        ''' 
        Euler-Chromer numerical integration. This function gets the
        acceleration from a potential function. In our case, this
        potential is Lennard-Jones. Based on the acceleration, it
        finds the velocity at the current timestep t using the 
        Euler-Chromer integration scheme. 
        
        Arguments:
        ----------
        t               {int}   :   Current timestep.
        potential       {func}  :   Inter-atomic potential (Lennard-Jones)
        '''
        a = potential(t)
        self.v[t+1] = self.v[t] + a * self.dt
        self.r[t+1] = self.r[t] + self.v[t+1] * self.dt
        
    def velocityVerlet(self, t, potential):
        ''' 
        Velocity-Verlet numerical integration. This function gets the
        acceleration from a potential function. In our case, this
        potential is Lennard-Jones. Based on the acceleration, it
        finds the velocity at the current timestep t using the 
        Velocity-Verlet integration scheme. 
        
        Arguments:
        ----------
        t               {int}   :   Current timestep.
        potential       {func}  :   Inter-atomic potential (Lennard-Jones)
        '''
        a = potential(t)
        self.r[t+1] = self.r[t] + self.v[t] * self.dt + 0.5 * a * self.dt**2
        a_new = potential(t+1)
        self.v[t+1] = self.v[t] + 0.5 * (a_new + a) * self.dt
        
    def dumpPositions(self, t, dumpfile):
        ''' 
        Dumping positions at timestep t to a dumpfile. We use the xyz-
        format, which can easily be visualized using Ovito.
        
        Arguments:
        ----------
        t               {int}   :   Current timestep.
        dumpfile        {str}   :   Name and address of dumpfile
        '''
        dat = np.column_stack((self.numparticles * ['Ar'], self.r[t]))
        np.savetxt(dumpfile, dat, header="{}\ntype x y z".format(self.numparticles), fmt="%s", comments='')
    
    def simulate(self, potential, integrator, dumpfile=None):
        ''' 
        Integration loop. Computes the time-development of position and 
        velocity using a given integrator and inter-atomic potential.
        
        Arguments:
        ----------
        potential       {func}  : Function defining the inter-atomic potential
        integrator      {func}  : Function defining the integrator
        dumpfile        {str}   : Filename that all the positions should be
                                  dumped to. If not specified, positions are
                                  not dumped.  
        '''
        if dumpfile is not None: 
            f=open(dumpfile,'ab')       # Open dumpfile
            self.dumpPositions(0,f)     # Dump initial positions
        from tqdm import tqdm
        for t in tqdm(range(self.N)):   # Integration loop
            integrator(t, potential)    # integrate to find velocities and positions
            if dumpfile is not None: self.dumpPositions(t+1,f) # dump positions to file
        if dumpfile is not None: f.close()      # Close dumpfile
        self.calculateDistanceMatrix(self.N)    # Calculate final distance 
        
    def plot_distance(self):
        ''' 
        Plot distance between all particles. The plot will contain a 
        graph for each particle pair, giving N(N-1)/2 graphs. It is 
        recommended to use just for a small number of particles.
        '''
        distance = np.sqrt(self.d)
        for i in range(self.numparticles):
            for j in range(i):
                plt.plot(self.time, distance[:,i,j], label="$i={}$, $j={}$".format(i,j))
        plt.legend(loc="best", fontsize=self.size)
        plt.xlabel(r"Time [$t'/\tau$]", **self.label_size)
        plt.ylabel("$r_{ij}$", **self.label_size)
        plt.show()
        
    def plot_energy(self):
        ''' 
        This function plots the kinetic, potential and total energy.
        The kinetic energy is taken from the kineticEnergy function,
        while the potential energy is taken from the specified potential
        (which in our case is Lennard-Jones).
        '''
        k = self.kineticEnergy()        # Kinetic energy
        p = self.lennardJonesEnergy()   # Potential energy
        e = k + p                       # Total energy
        plt.plot(self.time, k, label="Kinetic")
        plt.plot(self.time, p, label="Potential")
        plt.plot(self.time, e, label="Total energy")
        plt.legend(loc="best", fontsize=self.size)
        plt.xlabel(r"Time [$t'/\tau$]", **self.label_size)
        plt.ylabel(r"Energy [$\varepsilon$]", **self.label_size)
        plt.show()

if __name__ == "__main__":
    # EXAMPLE: TWO PARTICLES IN ONE DIMENSION
    simulator = MolecularDynamics(positions=[[0.0], [1.5]], T=5, dt=0.01)
    simulator.simulate(potential=obj.lennardJones, integrator=obj.eulerChromer)
    simulator.plot_distance()
