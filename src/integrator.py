class Integrator:
    def __init__(self):
        pass
        
    def __call__(self, r, v, a):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))
                 
    @staticmethod                  
    def boundary(r, v, lenbox, numdimensions, boundaries, potential):
        import numpy as np
        # Set boundaries
        for d in range(numdimensions):
            lo, hi = 0, lenbox
            if boundaries[d] == 'o':
                # Open boundaries: no action
                continue
            elif boundaries[d] == 'r':
                # Reflective boundaries
                r_d = r[:,d]
                v_d = v[:,d]
                # Reverse speed when hitting wall
                v_d = np.where(r_d>hi, -v_d, v_d)
                v_d = np.where(r_d<lo, -v_d, v_d)
                # Ensure that the particle is located inside box 
                r_d = np.where(r_d>hi, 2*hi - r_d, r_d)
                r_d = np.where(r_d<lo, 2*lo - r_d, r_d)
                v[:,d] = v_d
                r[:,d] = r_d
            elif boundaries[d] == 'p':
                # Periodic boundaries: do not affect speed
                r_d = r[:,d]
                # Move particle to other side of box when hitting wall
                r_d = np.where(r_d>hi, r_d - hi + lo, r_d)
                r_d = np.where(r_d<lo, r_d + hi + lo, r_d)
                r[:,d] = r_d
                # 
                zero = np.zeros(numdimensions)
                zero[d] = lo-hi
                r_dl = np.add(r, zero)
                zero = np.zeros(numdimensions)
                zero[d] = hi-lo
                r_dr = np.add(r, zero)
                
                
        a, u, d = potential(r)
        return r, v, a, u, d

class ForwardEuler(Integrator):
    def __init__(self, solver):
        self.solver = solver
        self.dt = solver.dt
        self.boundaries = solver.boundaries
        self.numdimensions = solver.numdimensions
        self.lenbox = solver.lenbox
        
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
        r, v, a, u, d = self.boundary(r, v, self.lenbox, self.numdimensions, self.boundaries, self.solver.potential)
        return r, v, a, u, d
        
class EulerChromer(Integrator):
    def __init__(self, solver):
        self.solver = solver
        self.dt = solver.dt
        self.boundaries = solver.boundaries
        self.numdimensions = solver.numdimensions
        self.lenbox = solver.lenbox
        
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
        r, v, a, u, d = self.boundary(r, v, self.lenbox, self.numdimensions, self.boundaries, self.solver.potential)
        return r, v, a, u, d

class VelocityVerlet(Integrator):
    def __init__(self, solver):
        self.solver = solver
        self.dt = solver.dt
        self.boundaries = solver.boundaries
        self.numdimensions = solver.numdimensions
        self.lenbox = solver.lenbox
        
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
        r, v, a_new, u, d = self.boundary(r, v, self.lenbox, self.numdimensions, self.boundaries, self.solver.potential)
        v += 0.5 * (a_new + a) * self.dt
        return r, v, a_new, u, d
