import numpy as np

class Boundaries:
    """ Boundary condition class. Ensures that the positions, velocities
    and forces act according to the desired boundary condition.
    """
    def __init__(self):
        pass
    
    def checkPosition(self, r):
        raise NotImplementedError ("Class {} has no instance 'checkPosition'."
                                   .format(self.__class__.__name__))
                                   
class Open(Boundaries):
    """ Open boundary conditions. Does not alter positions, velocities or
    forces.
    """
    def __init__(self):
        pass
        
    @staticmethod
    def __repr__():
        return "Open boundaries"
        
    @staticmethod
    def checkPosition(r):
        """ Check if the positions satisfy the boundary conditions.
        
        Parameters
        ----------
        r : ndarray
            current position array
            
        Returns
        -------
        ndarray
            changed position array
        """
        return r
        
    @staticmethod
    def checkVelocity(v):
        """ Check if the velocities satisfy the boundary conditions.
        
        Parameters
        ----------
        v : ndarray
            current velocity array
            
        Returns
        -------
        ndarray
            changed velocity array
        """
        return v
        
    @staticmethod
    def checkDistance(dr):
        """ Check if the distance vectors satisfy the boundary conditions.
        
        Parameters
        ----------
        dr : ndarray
            current distance vectors
            
        Returns
        -------
        ndarray
            changed distance vectors
        """
        return dr
        
class Reflective(Boundaries):
    def __init__(self, lenbox):
        self.lenbox = lenbox
        
    def __repr__(self):
        return "Reflective boundaries with box length {}".format(self.lenbox)
        
    def checkPosition(self, r):
        """ Check if the positions satisfy the boundary conditions.
        
        Parameters
        ----------
        r : ndarray
            current position array
            
        Returns
        -------
        ndarray
            changed position array
        """
        self.r = r
        r = np.where(r>self.lenbox, 2*self.lenbox - r, r)
        r = np.where(r<0, - r, r)
        return r
        
    def checkVelocity(self, v):
        """ Check if the velocities satisfy the boundary conditions.
        
        Parameters
        ----------
        v : ndarray
            current velocity array
            
        Returns
        -------
        ndarray
            changed velocity array
        """
        return np.where(self.r//self.lenbox == 0, v, -v)
        
    @staticmethod
    def checkDistance(dr):
        """ Check if the distance vectors satisfy the boundary conditions.
        
        Parameters
        ----------
        dr : ndarray
            current distance vectors
            
        Returns
        -------
        ndarray
            changed distance vectors
        """
        return dr
        
class Periodic(Boundaries):
    def __init__(self, lenbox):
        self.lenbox = lenbox
        
    def __repr__(self):
        return "Periodic boundaries with box length {}".format(self.lenbox)
        
    def checkPosition(self, r):
        """ Check if the positions satisfy the boundary conditions.
        
        Parameters
        ----------
        r : ndarray
            current position array
            
        Returns
        -------
        ndarray
            changed position array
        """
        r -= np.floor(self.lenbox/r) * self.lenbox
        return r
        
    @staticmethod
    def checkVelocity(v):
        """ Check if the velocities satisfy the boundary conditions.
        
        Parameters
        ----------
        v : ndarray
            current velocity array
            
        Returns
        -------
        ndarray
            changed velocity array
        """
        return v
        
    def checkDistance(self, dr):
        """ Check if the distance vectors satisfy the boundary conditions.
        
        Parameters
        ----------
        dr : ndarray
            current distance vectors
            
        Returns
        -------
        ndarray
            changed distance vectors
        """
        dr -= np.round(dr/self.lenbox) * self.lenbox
        return dr
