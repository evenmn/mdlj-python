import numpy as np

class Boundaries:
    """ Boundary condition class. Ensures that the positions, velocities
    and forces act according to the desired boundary condition.
    """
    def __init__(self):
        pass
    
    def correctPosition(self, r):
        raise NotImplementedError ("Class {} has no instance 'correctPosition'."
                                   .format(self.__class__.__name__))
                                   
    def correctVelocity(self, r):
        raise NotImplementedError ("Class {} has no instance 'correctVelocity'."
                                   .format(self.__class__.__name__))
    
    def correctDistance(self, r):
        raise NotImplementedError ("Class {} has no instance 'correctDistance'."
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
    def correctPosition(r):
        """ Check if the positions satisfy the boundary conditions.
        
        Parameters
        ----------
        r : ndarray
            current position array
            
        Returns
        -------
        ndarray
            position array correction
        """
        return np.zeros(r.shape)
        
    @staticmethod
    def correctVelocity(v):
        """ Check if the velocities satisfy the boundary conditions.
        
        Parameters
        ----------
        v : ndarray
            current velocity array
            
        Returns
        -------
        ndarray
            velocity correction
        """
        return np.zeros(v.shape)
        
    @staticmethod
    def correctDistance(dr):
        """ Check if the distance vectors satisfy the boundary conditions.
        
        Parameters
        ----------
        dr : ndarray
            current distance vectors
            
        Returns
        -------
        ndarray
            correction of distance vectors
        """
        return np.zeros(dr.shape)
        
class Reflective(Boundaries):
    def __init__(self, lenbox):
        self.lenbox = lenbox
        
    def __repr__(self):
        return "Reflective boundaries with box length {}".format(self.lenbox)
        
    def correctPosition(self, r):
        """ Check if the positions satisfy the boundary conditions.
        
        Parameters
        ----------
        r : ndarray
            current position array
            
        Returns
        -------
        ndarray
            position correction
        """
        self.r = r
        #passed_wall = np.floor(r/self.lenbox)
        #return - passed_wall * (2*r + (passed_wall+1) * 2*self.lenbox)
        
        #r = np.where(r>self.lenbox, 2*self.lenbox - r, r)
        #r = np.where(r<0, - r, r)
        #return r
        
        c = np.where(r>self.lenbox, -2*(r-self.lenbox), 0)
        c += np.where(r<0, -2*r, 0)
        return c
        
    def correctVelocity(self, v):
        """ Check if the velocities satisfy the boundary conditions.
        
        Parameters
        ----------
        v : ndarray
            current velocity array
            
        Returns
        -------
        ndarray
            velocity correction
        """
        #return - 2 * np.floor(self.r/self.lenbox) * v
        return np.where(self.r // self.lenbox != 0, -2*v, 0)
        #c += np.where(self.r<0, 
        #return np.where(self.r // self.lenbox == 0, v, -v)
        
    @staticmethod
    def correctDistance(dr):
        """ Check if the distance vectors satisfy the boundary conditions.
        
        Parameters
        ----------
        dr : ndarray
            current distance vectors
            
        Returns
        -------
        ndarray
            correction of distance vectors
        """
        return np.zeros(dr.shape)
        
class Periodic(Boundaries):
    def __init__(self, lenbox):
        self.lenbox = lenbox
        
    def __repr__(self):
        return "Periodic boundaries with box length {}".format(self.lenbox)
        
    def correctPosition(self, r):
        """ Check if the positions satisfy the boundary conditions.
        
        Parameters
        ----------
        r : ndarray
            current position array
            
        Returns
        -------
        ndarray
            position correction
        """
        return - np.floor(r/self.lenbox) * self.lenbox
        
    @staticmethod
    def correctVelocity(v):
        """ Check if the velocities satisfy the boundary conditions.
        
        Parameters
        ----------
        v : ndarray
            current velocity array
            
        Returns
        -------
        ndarray
            velocity correction
        """
        return np.zeros(v.shape)
        
    def correctDistance(self, dr):
        """ Check if the distance vectors satisfy the boundary conditions.
        
        Parameters
        ----------
        dr : ndarray
            current distance vectors
            
        Returns
        -------
        ndarray
            correction of distance vectors
        """
        
        return - np.round(dr/self.lenbox) * self.lenbox
