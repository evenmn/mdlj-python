class Thermo:
    def __init__(self, freq, file, quantities):
        self.freq = freq
        self.quantities = quantities
        self.f = open(file, 'w')
        self.make_header(self.f, quantities)

    @staticmethod
    def make_header(f, quantities):
        header = ""
        for quantity in quantities:
            header += "{:<12}".format(quantity)
        # header = " ".join(quantities)
        f.write(header + "\n")
        print("\n" + header)

    def collect_data(self, solver, quantities):
        # temporary way to set particle types
        # dat = []
        string = ""
        for quantity in quantities:
            string += "{:<12.3f}".format(getattr(self, quantity)(solver))
            # dat.append(getattr(self, quantity)(solver))
        # return ' '.join(map(str, dat))
        return string

    def __call__(self, solver):
        if solver.t % self.freq == 0:
            dat = self.collect_data(solver, self.quantities)
            self.f.write(dat + "\n")
            print(dat)

    @staticmethod
    def step(solver):
        return solver.t

    @staticmethod
    def time(solver):
        return solver.t * solver.dt

    @staticmethod
    def atoms(solver):
        return solver.numparticles

    @staticmethod
    def temp(solver):
        return (solver.v**2).sum() / (solver.numparticles * solver.numdimensions)

    @staticmethod
    def poteng(solver):
        return solver.u

    @staticmethod
    def kineng(solver):
        return (solver.v**2).sum()/2

    @staticmethod
    def velcorr(solver):
        if solver.t == solver.t0:
            solver.v0 = solver.v
            solver.v02 = solver.v0**2
        return ((solver.v * solver.v0) / solver.v02).sum() / solver.numparticles

    @staticmethod
    def mse(solver):
        if solver.t == solver.t0:
            solver.r0 = solver.r
        r = solver.r + solver.n * solver.boundaries.lenbox
        return ((r - solver.r0)**2).sum() / solver.numparticles

    def __del__(self):
        self.f.close()
