class Thermo:
    def __init__(self, freq, file, quantities):
        self.freq = freq
        self.quantities = quantities
        self.f = open(file, 'w')
        self.make_header(self.f, quantities)

    @staticmethod
    def make_header(f, quantities):
        header = " ".join(quantities)
        f.write(header + "\n")
        print("\n" + header)

    def collect_data(self, solver, quantities):
        # temporary way to set particle types
        dat = []
        for quantity in quantities:
            dat.append(getattr(self, quantity)(solver))
        return ' '.join(map(str, dat))

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
    def poteng(solver):
        return solver.u

    @staticmethod
    def kineng(solver):
        return (solver.v**2).sum()/2

    def __del__(self):
        self.f.close()
