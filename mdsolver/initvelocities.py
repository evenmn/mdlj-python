import numpy as np

class InitVelocities:
    """ Initial velocities class. Set the initial velocities according 
    to some method. 
    """
    def __init__(self):
        pass
        
    def __call__(self):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))
                                   
class SetVelocities(InitVelocities):
    """ Specify the velocities manually using a nested list. By using
    this method, the user has done all the work and the class will 
    just return the user input.
    
    Parameters
    ----------
    positions : array_like
        initial positions of all the particles. With this array, the user
        also specifies the number of particles and number of dimensions.
    """
    def __init__(self, velocities):
        self.velocities = velocities
        
    def __call__(self, par, dim):
        """ Get the velocities.
        
        Parameters
        ----------
        par : int
            number of particles
        dim : int
            number of dimensions
        
        Returns
        -------
        ndarray
            initial velocity configuration
        """
        assert len(self.velocities) == par, \
               "Number of velocities needs to match number of particles"
        assert len(self.velocities[0]) == dim, \
               "Velocity dim needs to match particle dim"
        return self.velocities
        
class Zero(InitVelocities):
    """ No initial velocities.
    """
    def __init__(self):
        pass
    
    def __call__(self, par, dim):
        """ Get the velocities.
        
        Parameters
        ----------
        par : int
            number of particles
        dim : int
            number of dimensions
        
        Returns
        -------
        ndarray
            initial velocity configuration
        """
        return np.zeros((par, dim))
        
class Gauss(InitVelocities):
    """ Gaussian distributed initial velocities.
    
    Parameters
    ----------
    mean : float
        mean value of Gaussian distribution
    var : float
        variance of Gaussian distribution
    """
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        
    def __call__(self, par, dim):
        """ Get the velocities.
        
        Parameters
        ----------
        par : int
            number of particles
        dim : int
            number of dimensions
        
        Returns
        -------
        ndarray
            initial velocity configuration
        """
        return np.random.normal(self.mean, self.var, size=(par, dim)) 
        
class Temperature(InitVelocities):
    """ Set the velocities to get a certain initial temperature
    specified by T. Using the formula
        T = v^2/ND
    inversely.
        
    Parameters
    ----------
    T : float
        initial temperature given in Kelvin
    """
    def __init__(self, T):
        self.T = T/119.7        # Transform from Kelvin to reduced units
        
    def __call__(self, par, dim):
        """ Get the velocities.
        
        Parameters
        ----------
        par : int
            number of particles
        dim : int
            number of dimensions
        
        Returns
        -------
        ndarray
            initial velocity configuration
        """
        return np.random.normal(0, np.sqrt(self.T), size=(par, dim))
