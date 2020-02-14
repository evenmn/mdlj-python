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
            self.v = np.zeros((self.N+1, self.numparticles, numdimensions))
        elif velocity=="gauss":
            self.v = np.random.normal(0, 1, size=(self.N+1, numparticles, numdimensions))
        elif type(velocity) == list:
            self.v[0] = velocity

        # declare distance matrix
        self.d = np.zeros((self.N+1, self.numparticles, self.numparticles))
        
        # for plotting
        self.size = size                        # Label size in plots
        self.label_size = {"size":str(size)}    # Dictionary with size
        plt.style.use("bmh")                    # Beautiful plots
        plt.rcParams["font.family"] = "Serif"   # Font
        
    def face_centered_cube(self, n, d, dim):
        '''
        Creating a face-centered cube of n^dim unit cells with
        4 particles in each unit cell. The number of particles
        then becomes (dim+1) * n ^ dim. Each unit cell has a 
        length d.
        
        Arguments:
        ----------
        n       {int}   :   Number of unit cells in each dimension
        d       {float} :   Length of a unit cell
        dim     {int}   :   Number of dimensions
        '''
        self.numparticles = (dim+1) * n ** dim
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
            
    def calculateDistanceMatrix(self,t):
        ''' Compute the distance matrix at timestep t. '''
        x = self.r[t][:,np.newaxis,:]
        y = self.r[t][np.newaxis,:,:]
        dr = x - y
        del x, y
        self.d[t] = np.einsum('ijk,ijk->ij',dr,dr)
        return dr
        
    def lennardJones(self,t):
        ''' Lennard-Jones inter-atomic potential. '''
        dr = self.calculateDistanceMatrix(t)
        l = np.nan_to_num(self.d[t]**(-3))
        m = np.square(l)
        factor = np.divide(2 * m - l, self.d[t])
        del l, m
        factor[factor == np.inf] = 0
        return - 6 * np.einsum('ij,ijk->jk',factor,dr)
        del dr
        
    def lennardJonesEnergy(self):
        ''' Returns the potential energy from the Lennard-Jones potential. '''
        l = np.nan_to_num(self.d**(-3/2))
        m = np.square(l)
        u = m - l
        del l, m
        u[u == np.inf] = 0
        p = u.sum(axis=1).sum(axis=1)/2
        del u
        return p - np.min(p)
        
    def kineticEnergy(self):
        ''' Returns the total kinetic energy for each timestep. '''
        kPar = np.sum(np.square(self.v), axis=2)
        return np.sum(kPar, axis=1)/2
        del kPar
        
    def forwardEuler(self,t,potential):
        ''' Forward Euler integration. '''
        # calculate force acting on all the particles
        a = potential(t)
        self.v[t+1] = self.v[t] + a * self.dt
        self.r[t+1] = self.r[t] + self.v[t] * self.dt
        
    def eulerChromer(self,t,potential):
        ''' Euler-Chromer integration. '''
        a = potential(t)
        self.v[t+1] = self.v[t] + a * self.dt
        self.r[t+1] = self.r[t] + self.v[t+1] * self.dt
        
    def velocityVerlet(self,t,potential):
        ''' Velocity-Verlet integration. '''
        a = potential(t)
        self.r[t+1] = self.r[t] + self.v[t] * self.dt + 0.5 * a * self.dt**2
        a_new = potential(t+1)
        self.v[t+1] = self.v[t] + 0.5 * (a_new + a) * self.dt
        
    def dumpPositions(self, t, dumpfile):
        ''' Dump positions to file. '''
        dat = np.column_stack((self.numparticles * ['Ar'], self.r[t]))
        np.savetxt(dumpfile, dat, header="{}\ntype x y z".format(self.numparticles), fmt="%s", comments='')
    
    def simulate(self, potential, integrator, dumpfile=None):
        ''' Integration loop. 
        
        Arguments:
        ----------
        potential       {func}  : Function defining the inter-atomic potential
        integrator      {func}  : Function defining the integrator
        dumpfile        {str}   : Filename that all the positions shoudl be
                                  dumped to. If not specified, positions are
                                  not dumped.  
        '''
        
        if dumpfile is not None: 
            f=open(dumpfile,'ab')
            self.dumpPositions(0,f)
        
        from tqdm import tqdm
        for t in tqdm(range(self.N)):
            
            # integrate to find velocities and positions
            integrator(t, potential)
            
            # dump positions to file
            if dumpfile is not None: self.dumpPositions(t+1,f)
            
        if dumpfile is not None: f.close()
        self.calculateDistanceMatrix(self.N)
        
    def plot_distance(self):
        ''' Plot distance between all particles. '''
        for i in range(self.numparticles):
            for j in range(i):
                plt.plot(self.time, self.d[:,i,j], label="$i={}$, $j={}$".format(i,j))
        plt.legend(loc="best", fontsize=self.size)
        plt.xlabel(r"Time [$t'/\tau$]", **self.label_size)
        plt.ylabel("$r_{ij}$", **self.label_size)
        plt.show()
        
    def plot_energy(self):
        ''' Plot kinetic, potential and total energy. '''
        
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
    obj = MolecularDynamics(numparticles=2, numdimensions=3, T=5, dt=0.01)
    obj.initialize(position=[[0,0,0],[1.5,0,0]])
    obj.simulate(potential=obj.lennardJones, integrator=obj.eulerChromer)
    obj.plot_distance()
