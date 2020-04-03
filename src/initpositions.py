class InitPositions:
    def __init__(self):
        pass
        
    def __call__(self):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))
                                   
class setPositions(InitPositions):
    def __init__(self, positions):
        self.positions = positions
        
    def __call__(self):
        return self.positions
        
class FCC(InitPositions):
    """ Creating a face-centered cube of n^dim unit cells with
    4 particles in each unit cell. The number of particles
    then becomes (dim+1) * n ^ dim. Each unit cell has a 
    length d. L=nd
    
    Parameters
    ----------
    cells : int
        number of unit cells in each dimension
    lenbox : float
        length of box
    dim : int
        number of dimensions
        
    Returns
    -------
    2darray
        initial particle configuration
    """
    def __init__(self, cells, lenbox, dim=3):
        self.cells = cells
        self.lenbox = lenbox
        self.dim = dim
    
    def __call__(self):
        from numpy import zeros
        par = (self.dim+1) * self.cells ** self.dim
        r = zeros((par, self.dim))
        counter = 0
        if self.dim==1:
            for i in range(self.cells):
                r[counter+0] = [i]
                r[counter+1] = [0.5+i]
                counter +=2
        elif self.dim==2:
            for i in range(self.cells):
                for j in range(self.cells):
                    r[counter+0] = [i, j]
                    r[counter+1] = [i, 0.5+j]
                    r[counter+2] = [0.5+i, j]
                    counter += 3
        elif self.dim==3:
            for i in range(self.cells):
                for j in range(self.cells):
                    for k in range(self.cells):
                        r[counter+0] = [i, j, k]
                        r[counter+1] = [i, 0.5+j, 0.5+k]
                        r[counter+2] = [0.5+i, j, 0.5+k]
                        r[counter+3] = [0.5+i, 0.5+j, k]
                        counter += 4
        else:
            raise ValueError("The number of dimensions needs to be in [1,3]")
        # Scale initial positions correctly
        r *= self.lenbox / self.cells
        return r
                                   
