class Integrator:
    def __init__(self):
        pass
        
    def __call__(self, r, v, a):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))

class ForwardEuler(Integrator):
    def __init__(self, solver):
        self.solver = solver
        self.dt = solver.dt
        
    def __call__(self, r, v, a):
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
        r += v * self.dt
        v += a * self.dt
        a, u, d = self.solver.potential(r)
        return r, v, a, u, d
        
class EulerChromer(Integrator):
    def __init__(self, solver):
        self.solver = solver
        self.dt = solver.dt
        
    def __call__(self, r, v, a):
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
        v += a * self.dt
        r += v * self.dt
        a, u, d = self.solver.potential(r)
        return r, v, a, u, d

class VelocityVerlet(Integrator):
    def __init__(self, solver):
        self.solver = solver
        self.dt = solver.dt
        
    def __call__(self, r, v, a):
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
        r += v * self.dt + 0.5 * a * self.dt**2
        a_new, u, d = self.solver.potential(r)
        v += 0.5 * (a_new + a) * self.dt
        return r, v, a_new, u, d
