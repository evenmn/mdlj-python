import numpy as np
class Potential:
    def __init__(self):
        pass
        
    def __call__(self, r):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))

class LennardJones(Potential):
    def __init__(self, cutoff=3.0):
        self.cutoffSqrd = cutoff * cutoff
        
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
        
    @staticmethod
    def potentialEnergy(u):
        """ Calculates the potential energy at timestep t, based on 
        the potential energies of all particles stored in the matrix
        u.
        
        Parameters
        u : ndarray
            array containing the potential energy of all the particles.
        """
        u[u == np.inf] = 0
        return 2 * np.sum(u)       # Multiply with 4 / 2
        
    def __call__(self, r):
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
        dr, d, l, m = self.calculateDistanceMatrix(r)
        l = np.where(d>self.cutoffSqrd, 0, l)
        m = np.where(d>self.cutoffSqrd, 0, m)
        u = self.potentialEnergy(m - l)
        factor = np.divide(2 * m - l, d)            # (2/r^12 - 1/r^6)/r^2
        factor[factor == np.inf] = 0
        force = - 24 * np.einsum('ij,ijk->jk',factor,dr)
        return force, u, d
