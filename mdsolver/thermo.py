import re


class Thermo:
    def __init__(self, freq, file, quantities):
        self.freq = freq
        self.quantities = quantities
        self.f = open(file, 'w')
        self.make_header()

    def make_header(self):
        self.header = ""
        for quantity in self.quantities:
            self.header += "{:<12}".format(quantity)
        self.f.write(self.header + "\n")

    def write_header(self):
        print("\n" + self.header)

    def collect_data(self, solver, quantities):
        string = ""
        for quantity in quantities:
            if "[" in quantity:
                label = quantity.split('[')[0]
                indices = re.findall(r"(?<!\.)\d+(?!\.)", quantity)
                string += "{:<12.3f}".format(getattr(self, label)(solver, *tuple(map(int, indices))))
            else:
                string += "{:<12.3f}".format(getattr(self, quantity)(solver))
        return string

    def __call__(self, solver):
        dat = ""
        if solver.t % self.freq == 0:
            dat = self.collect_data(solver, self.quantities) + "\n"
            self.f.write(dat)
        return dat

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

    @staticmethod
    def r(solver, i, j):
        return solver.r[i, j]

    @staticmethod
    def v(solver, i, j):
        return solver.v[i, j]

    @staticmethod
    def a(solver, i, j):
        return solver.a[i, j]

    def __del__(self):
        self.f.close()
