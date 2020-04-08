class Integrator:
    """ Integrator class. Takes a old state and returns a new state.
    """
    def __init__(self):
        pass
        
    def __call__(self, r, v, a):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))

class ForwardEuler(Integrator):
    """ Forward-Euler integrator, based on the integration scheme
        v[i+1] = v[i] + a * dt
        r[i+1] = r[i] + v[i] * dt
        
    Parameters
    ----------
    solver : obj
        class object defined by moleculardynamics.py. Takes the MDSolver 
        class as argument
    """
    def __init__(self, solver):
        self.solver = solver
        self.boundaries = solver.boundaries
        self.dt = solver.dt
        
        
    def __repr__(self):
        """ Representing the integrator.
        """
        return "Forward-Euler integrator"
        
    def __call__(self, r, v, a):
        """ This function calculated the new position and velocity based on 
        the integration scheme, and check if they satisfy the boundary
        conditions. Furthermore, the new acceleration is calculated.
        
        Parameters
        ----------
        r : ndarray
            previous position array
        v : ndarray
            previous velocity array
        a : ndarray
            previous acceleration
            
        Returns
        -------
        r : ndarray
            new position array
        v : ndarray
            new velocity array
        a : ndarray
            new acceleration array
        u : float
            potential energy of the new state
        d : ndarray
            distance matrix of the new state
        corrected_position : ndarray
            corrected position
        """
        r, v, a = r.copy(), v.copy(), a.copy()
        r += v * self.dt
        v += a * self.dt
        corrected_position = self.boundaries.correctPosition(r)
        r += corrected_position
        v += self.boundaries.correctVelocity(v)
        a, u, d = self.solver.potential(r)
        return r, v, a, u, d, corrected_position
        
class EulerChromer(Integrator):
    """ Euler-Chromer integrator, based on the integration scheme
        v[i+1] = v[i] + a * dt
        r[i+1] = r[i] + v[i+1] * dt
        
    Parameters
    ----------
    solver : obj
        class object defined by moleculardynamics.py. Takes the MDSolver 
        class as argument
    """
    def __init__(self, solver):
        self.solver = solver
        self.boundaries = solver.boundaries
        self.dt = solver.dt
        
    def __repr__(self):
        """ Representing the integrator.
        """
        return "Euler-Chromer integrator"
        
    def __call__(self, r, v, a):
        """ This function calculated the new position and velocity based on 
        the integration scheme, and check if they satisfy the boundary
        conditions. Furthermore, the new acceleration is calculated.
        
        Parameters
        ----------
        r : ndarray
            previous position array
        v : ndarray
            previous velocity array
        a : ndarray
            previous acceleration
            
        Returns
        -------
        r : ndarray
            new position array
        v : ndarray
            new velocity array
        a : ndarray
            new acceleration array
        u : float
            potential energy of the new state
        d : ndarray
            distance matrix of the new state
        corrected_position : ndarray
            corrected position
        """
        r, v, a = r.copy(), v.copy(), a.copy()
        v += a * self.dt
        r += v * self.dt
        corrected_position = self.boundaries.correctPosition(r)
        r += corrected_position
        v += self.boundaries.correctVelocity(v) 
        a, u, d = self.solver.potential(r)
        return r, v, a, u, d, corrected_position

class VelocityVerlet(Integrator):
    """ Velocity-Verlet integrator, based on the integration scheme
        r[i+1] = r[i] + v[i] * dt + 0.5 * a * dt^2
        v[i+1] = v[i] + 0.5 * (a + a_new) * dt
        
    Parameters
    ----------
    solver : obj
        class object defined by moleculardynamics.py. Takes the MDSolver 
        class as argument
    """
    def __init__(self, solver):
        self.solver = solver
        self.boundaries = solver.boundaries
        self.dt = solver.dt
        
    def __repr__(self):
        """ Representing the integrator.
        """
        return "VelocityVerlet integrator"
        
    def __call__(self, r, v, a):
        """ This function calculated the new position and velocity based on 
        the integration scheme, and check if they satisfy the boundary
        conditions. Furthermore, the new acceleration is calculated.
        
        Parameters
        ----------
        r : ndarray
            previous position array
        v : ndarray
            previous velocity array
        a : ndarray
            previous acceleration
            
        Returns
        -------
        r : ndarray
            new position array
        v : ndarray
            new velocity array
        a_new : ndarray
            new acceleration array
        u : float
            potential energy of the new state
        d : ndarray
            distance matrix of the new state
        corrected_position : ndarray
            corrected position
        """
        r, v, a = r.copy(), v.copy(), a.copy()
        r += v * self.dt + 0.5 * a * self.dt**2
        corrected_position = self.boundaries.correctPosition(r)
        r += corrected_position
        a_new, u, d = self.solver.potential(r)
        v += 0.5 * (a_new + a) * self.dt
        v += self.boundaries.correctVelocity(v)
        return r, v, a_new, u, d, corrected_position
