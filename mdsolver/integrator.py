class Integrator:
    """ Integrator class. Takes a old state and returns a new state.
    """
    def __init__(self):
        pass

    def __call__(self, r, v, a):
        raise NotImplementedError("Class {} has no instance '__call__'."
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
        self.boundary = solver.boundary
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
        """
        r, v, a = r.copy(), v.copy(), a.copy()
        r += v * self.dt
        v += a * self.dt
        r, n = self.boundary.checkPosition(r)
        v = self.boundary.checkVelocity(v)
        a, u = self.solver.potential(r, self.solver.compute_poteng)
        return r, n, v, a, u


class EulerCromer(Integrator):
    """ Euler-Cromer integrator, based on the integration scheme
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
        self.boundary = solver.boundary
        self.dt = solver.dt

    def __repr__(self):
        """ Representing the integrator.
        """
        return "Euler-Cromer integrator"

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
        """
        r, v, a = r.copy(), v.copy(), a.copy()
        v += a * self.dt
        r += v * self.dt
        r, n = self.boundary.checkPosition(r)
        v = self.boundary.checkVelocity(v)
        a, u = self.solver.potential(r, self.solver.compute_poteng)
        return r, n, v, a, u


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
        self.boundary = solver.boundary
        self.dt = solver.dt

    def __repr__(self):
        """ Representing the integrator.
        """
        return "Velocity Verlet integrator"

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
        """
        r, v, a = r.copy(), v.copy(), a.copy()
        r += v * self.dt + 0.5 * a * self.dt**2
        r, n = self.boundary.checkPosition(r)
        a_new, u = self.solver.potential(r, self.solver.compute_poteng)
        v += 0.5 * (a_new + a) * self.dt
        v = self.boundary.checkVelocity(v)
        return r, n, v, a_new, u
