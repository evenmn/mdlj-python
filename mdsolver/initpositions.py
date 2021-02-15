from numpy import asarray, zeros


class InitPosition:
    """ Initial position class. Set the initial positions according
    to some method.
    """
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError("Class {} has no instance '__call__'."
                                  .format(self.__class__.__name__))


class SetPosition(InitPosition):
    """ Specify the position manually using a nested list. By using
    this method, the user has done all the work and the class will
    just return the user input.

    Parameters
    ----------
    position : array_like
        initial positions of all the particles. With this array, the user
        also specifies the number of particles and number of dimensions.
    """
    def __init__(self, position):
        self.position = position

    def __call__(self):
        """ Get the initial position.

        Returns
        -------
        ndarray
            initial particle configuration
        """
        return asarray(self.position)


class FCC(InitPosition):
    """ Creating a face-centered cube of n^dim unit cells with
    4 particles in each unit cell. The number of particles
    then becomes (dim+1) * n ^ dim. Each unit cell has a
    length d. L=nd

    Parameters
    ----------
    cells : int
        number of unit cells in each dimension
    lenbulk : float
        length of box
    dim : int
        number of dimensions
    """
    def __init__(self, cells, lenbulk, dim=3):
        self.cells = cells
        self.lenbulk = lenbulk
        self.dim = dim

    def __call__(self):
        """ Get the initial position.

        Returns
        -------
        ndarray
            initial particle configuration
        """
        par = (self.dim+1) * self.cells ** self.dim
        r = zeros((par, self.dim))
        counter = 0
        if self.dim == 1:
            for i in range(self.cells):
                r[counter+0] = [i]
                r[counter+1] = [0.5+i]
                counter += 2
        elif self.dim == 2:
            for i in range(self.cells):
                for j in range(self.cells):
                    r[counter+0] = [i, j]
                    r[counter+1] = [i, 0.5+j]
                    r[counter+2] = [0.5+i, j]
                    counter += 3
        elif self.dim == 3:
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
        r *= self.lenbulk / self.cells
        return r


class Restart(InitPosition):
    """Restart from a already written XYZ-file
    """
    def __init__(self, filename):
        if hasattr(filename, "read"):
            xyzfile = filename
        else:
            xyzfile = open(filename, 'r')
        self.read_file_to_dataframe(xyzfile)

    def read_xyz(self, xyzfile):
        """Read XYZ file and store contents in a Pandas DataFrame
        """
        # first line contains number of atoms
        self.numatom = int(xyzfile.readline().split()[0])
        # second line contains a comment
        self.comment = xyzfile.readline()[:-3]
        # rest of the lines contain coordinates structured Element X Y Z
        string = "Element X Y Z \n" + xyzfile.read()
        self.contents = pd.read_table(StringIO(string), sep=r'\s+')

    def __call__(self):
        return self.contents.to_numpy()
