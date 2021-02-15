import numpy as np


class InitVelocity:
    """ Initial velocities class. Set the initial velocities according
    to some method.
    """
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError("Class {} has no instance '__call__'."
                                  .format(self.__class__.__name__))


class SetVelocity(InitVelocity):
    """ Specify the velocity manually using a nested list. By using
    this method, the user has done all the work and the class will
    just return the user input.

    Parameters
    ----------
    positions : array_like
        initial positions of all the particles. With this array, the user
        also specifies the number of particles and number of dimensions.
    """
    def __init__(self, velocity):
        self.velocity = velocity

    def __call__(self, par, dim):
        """ Get the velocity.

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
        assert len(self.velocity) == par, \
               "Number of velocities needs to match number of particles"
        assert len(self.velocity[0]) == dim, \
               "Velocity dim needs to match particle dim"
        return np.asarray(self.velocity)


class Zero(InitVelocity):
    """ No initial velocity.
    """
    def __init__(self):
        pass

    def __call__(self, par, dim):
        """ Get the velocity.

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


class Gauss(InitVelocity):
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
        """ Get the velocity.

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


class Temperature(Gauss):
    """ Set the velocity to get a certain initial temperature
    specified by T. Using the formula
        T = <v>^2/ND
    inversely.

    Parameters
    ----------
    T : float
        initial temperature given in Kelvin
    """
    def __init__(self, T):
        self.mean = 0
        self.var = np.sqrt(T)
