import numpy as np

class InitVelocities:
    def __init__(self):
        pass
        
    def __call__(self):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))
                                   
class setVelocities(InitVelocities):
    def __init__(self, velocity):
        self.velocity = velocity
        
    def __call__(self, par, dim):
        assert self.velocity.shape[0] == par ("Number of velocities needs to match number of particles")
        assert self.velocity.shape[1] == dim ("Velocity dim needs to match particle dim")
        return self.velocity
        
class Zero(InitVelocities):
    def __init__(self):
        pass
    
    def __call__(self, par, dim):
        return np.zeros((par, dim))
        
class Gauss(InitVelocities):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        
    def __call__(self, par, dim):
        return np.random.normal(self.mean, self.var, size=(par, dim)) 
        
class Temperature(InitVelocities):
    def __init__(self, T):
        self.T = T/119.7        # Transform from Kelvin to reduced units
        
    def __call__(self, par, dim):
        return np.random.normal(0, np.sqrt(self.T), size=(par, dim))
