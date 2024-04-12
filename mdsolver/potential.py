import numpy as np


class Potential:
    """ Potential class. Find the force acting on the particles
    given a potential.
    """
    def __init__(self):
        pass

    def __call__(self, r):
        raise NotImplementedError("Class {} has no instance '__call__'."
                                  .format(self.__class__.__name__))

    def potentialEnergy(self, u, cutoff):
        raise NotImplementedError("Class {} has no instance 'potentialEnergy'."
                                  .format(self.__class__.__name__))


class LennardJones(Potential):
    """ The Lennard-Jones potential. Taking the form
        U(r) = 4ε((σ/r)^12 - (σ/r)^6)

    Parameters
    ----------
    solver : obj
        class object defined by moleculardynamics.py. Takes the MDSolver
        class as argument
    cutoff : float
        cutoff distance: maximum length of the interactions. 3 by default.
    """
    def __init__(self, solver, cutoff=3):
        self.cutoff = cutoff
        self.cutoffSqrd = cutoff * cutoff
        self.boundary = solver.boundary

        # Generate indices of upper and lower triangles
        par = solver.numparticles
        dim = solver.numdimensions
        self.forceShell = np.zeros((par, par, dim))
        self.upperTri = np.triu_indices(par, 1)
        self.index = np.array(self.upperTri).T
        self.shift = par * (cutoff**(-12) - cutoff**(-6))

    def __repr__(self):
        """ Representing the potential.
        """
        return "Lennard-Jones potential"

    def set_boundary(self, boundary):
        self.boundary = boundary

    def calculateDistanceMatrix(self, r):
        """ Compute the distance matrix (squared) at timestep t. In the
        integration loop, we only need the distance squared, which
        means that we do not need to take the square-root of the
        distance. We also exploit Newton's third law and calculate the
        needed forces just once. Additionally, we only care about the
        particles within a distance specified by the cutoff distance.

        Parameters
        ----------
        r : ndarray
            spatial coordinates at some timestep

        Returns
        -------
        distanceSqrdAll : ndarray
            distance between all particles squared
        distanceSqrd : ndarray
            distance between particles that are closer than the cutoff
        dr : ndarray
            distance vector between particles that are closer than the cutoff
        indices : ndarray
            indices of the distance components that are closer than cutoff
        """
        # Find distance vector matrix and distance matrix
        x, y = r[:, np.newaxis, :], r[np.newaxis, :, :]
        drAll = x - y                                 # distance vector matrix
        drAll = self.boundary.checkDistance(drAll)  # check if satisfy bc
        distanceSqrdAll = np.einsum('ijk,ijk->ij', drAll, drAll)    # r^2

        # Pick the upper triangular elements only from the matrices and flatten
        distanceSqrdHalf = distanceSqrdAll[self.upperTri]
        drHalf = drAll[self.upperTri]

        # Pick the components that are closer than the cutoff distance only
        indices = np.nonzero(distanceSqrdHalf<self.cutoffSqrd)
        distanceSqrd = distanceSqrdHalf[indices]
        dr = drHalf[indices]
        return distanceSqrdAll, distanceSqrd, dr, indices

    @staticmethod
    def potentialEnergy(u, shift):
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
        return 4 * (np.sum(u) - shift)

    def __call__(self, r, return_energy=False):
        """ Lennard-Jones inter-atomic force. This is used in the
        integration loop to calculate the acceleration of particles.

        Parameters
        ----------
        r : ndarray
            spatial coordinates at some timestep

        Returns
        -------
        ndarray
            the netto force acting on every particle
        float
            total potential energy
        ndarray
            current distance matrix
        """
        # Compute force between particles closer than cutoff
        distanceSqrdAll, distanceSqrd, dr, indices = self.calculateDistanceMatrix(r)
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))      # 1/r^6
        distancePowTwelveInv = distancePowSixInv**2                # 1/r^12
        factor = np.divide(2 * distancePowTwelveInv - distancePowSixInv, distanceSqrd)            # (2/r^12 - 1/r^6)/r^2
        factor[factor == np.inf] = 0
        force = 24 * np.einsum('i,ij->ij', factor, dr)

        # Connect forces to correct particles
        forceMatrix = self.forceShell.copy()
        index = self.index[indices].T
        forceMatrix[(index[0], index[1])] = force
        forceMatrix[(index[1], index[0])] = -force

        # Return net force on each particle and potetial energy
        forceParticles = np.sum(forceMatrix, axis=1)
        if return_energy:
            u = self.potentialEnergy(distancePowTwelveInv - distancePowSixInv, self.shift)
        else:
            u = None
        return forceParticles, u
