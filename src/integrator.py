class Integrator:
    def __init__(self):
        pass
        
    def __call__(self, r, v, a):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))
                 
    @staticmethod                  
    def boundary(r, v, lenbox, numdimensions, boundary, potential):
        import numpy as np
        # Set boundaries
        hi, lo = lenbox, 0
        if boundary == 'o':
            # Open boundaries: no action
            pass
        elif boundary == 'r':
            # Reflective boundaries
            # Reverse speed when hitting wall
            v = np.where(r>hi, -v, v)
            v = np.where(r<lo, -v, v)
            
            # Ensure that the particle is located inside box 
            r = np.where(r>hi, 2*hi - r, r)
            r = np.where(r<lo, 2*lo - r, r)
        elif boundary == 'p':
            r -= np.floor(lenbox/r) * lenbox
            
        a, u, d = potential(r) 
        return r, v, a, u, d

class ForwardEuler(Integrator):
    def __init__(self, solver):
        self.solver = solver
        self.dt = solver.dt
        self.boundaryy = solver.boundary
        self.numdimensions = solver.numdimensions
        self.lenbox = solver.lenbox
        
    def __repr__(self):
        """ Representing the integrator.
        """
        return "Forward-Euler integrator"
        
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
        r, v, a, u, d = self.boundary(r, v, self.lenbox, self.numdimensions, self.boundaryy, self.solver.potential)
        return r, v, a, u, d
        
class EulerChromer(Integrator):
    def __init__(self, solver):
        self.solver = solver
        self.dt = solver.dt
        self.boundaryy = solver.boundary
        self.numdimensions = solver.numdimensions
        self.lenbox = solver.lenbox
        
    def __repr__(self):
        """ Representing the integrator.
        """
        return "Euler-Chromer integrator"
        
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
        r, v, a, u, d = self.boundary(r, v, self.lenbox, self.numdimensions, self.boundaryy, self.solver.potential)
        return r, v, a, u, d

class VelocityVerlet(Integrator):
    def __init__(self, solver):
        self.solver = solver
        self.dt = solver.dt
        self.boundaryy = solver.boundary
        self.numdimensions = solver.numdimensions
        self.lenbox = solver.lenbox
        
    def __repr__(self):
        """ Representing the integrator.
        """
        return "VelocityVerlet integrator"
        
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
        r, v, a_new, u, d = self.boundary(r, v, self.lenbox, self.numdimensions, self.boundaryy, self.solver.potential)
        v += 0.5 * (a_new + a) * self.dt
        return r, v, a_new, u, d
