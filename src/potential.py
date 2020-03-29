import numpy as np
class Potential:
    def __init__(self):
        pass
        
    def __call__(self, r):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))
                     
    @staticmethod              
    def potentialEnergy(u):
        raise NotImplementedError ("Class {} has no instance 'potentialEnergy'."
                                   .format(self.__class__.__name__))

class LennardJones(Potential):
    def __init__(self, cutoff=3.0):
        self.cutoffSqrd = cutoff * cutoff
        
    #@staticmethod
    def calculateDistanceMatrix(self, r):
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
        par, dim = r.shape
        half = par*par // 2
        x, y = r[:,np.newaxis,:], r[np.newaxis,:,:]
        drAll = x - y
        distanceSqrdAll = np.einsum('ijk,ijk->ij',drAll,drAll)        # r^2
        upperTri = np.triu_indices(par, 1)
        distanceSqrdHalf = distanceSqrdAll[upperTri]
        drHalf = drAll[upperTri]
        indices = np.nonzero(distanceSqrdHalf<self.cutoffSqrd)
        distanceSqrd = distanceSqrdHalf[indices]      # Ignoring the particles separated
        dr = drHalf[indices]                          # by a distance > cutoff
        a = np.arange(par*par).reshape(par,par)
        b = a[upperTri]
        c = b[indices]
        return distanceSqrdAll, distanceSqrd, dr, c
        
    @staticmethod
    def potentialEnergy(u):
        """ Calculates the potential energy at timestep t, based on 
        the potential energies of all particles stored in the matrix
        u.
        
        Parameters
        u : ndarray
            array containing the potential energy of all the particles.
        """
        # TODO: Shift potential
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
        par, dim = r.shape
        distanceSqrdAll, distanceSqrd, dr, indices = self.calculateDistanceMatrix(r)
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))      # 1/r^6
        distancePowTwelveInv = distancePowSixInv**2                # 1/r^12
        factor = np.divide(2 * distancePowTwelveInv - distancePowSixInv, distanceSqrd)            # (2/r^12 - 1/r^6)/r^2
        factor[factor == np.inf] = 0
        #print(factor)
        #print(dr)
        force = - 24 * np.einsum('i,ij->ij',factor,dr)
        #print(force)
        #print(indices)
        force2 = np.zeros((par*par,dim))
        force2[indices] = -force
        force2[::-1][indices] = force
        force2 = force2.reshape(par,par,dim)
        force3 = np.sum(force2, axis=1)
        #print(force3)
        #print("")
        u = self.potentialEnergy(distancePowTwelveInv - distancePowSixInv)
        return force3, u, distanceSqrdAll
