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

    def __call__(self, shape):
        """ Get the velocity.

        Parameters
        ----------
        shape: tuple
            shape of position matrix (par, dim)

        Returns
        -------
        ndarray
            initial velocity configuration
        """
        velocity = np.asarray(self.velocity)
        assert velocity.shape == shape
        return velocity


class Zero(InitVelocity):
    """ No initial velocity.
    """
    def __init__(self):
        pass

    def __call__(self, shape):
        """ Get the velocity.

        Parameters
        ----------
        shape: tuple
            shape of position matrix (par, dim)

        Returns
        -------
        ndarray
            initial velocity configuration
        """
        return np.zeros(shape)


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

    def __call__(self, shape):
        """ Get the velocity.

        Parameters
        ----------
        shape: tuple
            shape of position matrix (par, dim)

        Returns
        -------
        ndarray
            initial velocity configuration
        """
        return np.random.normal(self.mean, self.var, size=shape)


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
