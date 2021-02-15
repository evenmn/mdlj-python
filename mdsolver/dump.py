from numpy import savetxt, column_stack


class Dump:
    def __init__(self, freq, file, quantities):
        self.freq = freq
        self.quantities = quantities
        self.f = open(file, 'w')

    def collect_data(self, solver, quantities):
        # temporary way to set particle types
        dat = [solver.numparticles * ['Ar']]
        for quantity in quantities:
            dat.append(getattr(self, quantity)(solver))
        return column_stack(dat)

    @staticmethod
    def make_header(numparticles, quantities):
        header = f"{numparticles}\ntype "
        header += " ".join(quantities)
        return header

    def __call__(self, solver):
        if solver.t % self.freq == 0:
            dat = self.collect_data(solver, self.quantities)
            header = self.make_header(solver.numparticles, self.quantities)
            savetxt(self.f, dat, header=header, fmt="%s", comments='')

    @staticmethod
    def x(solver):
        return solver.r[:, 0]

    @staticmethod
    def y(solver):
        return solver.r[:, 1]

    @staticmethod
    def z(solver):
        return solver.r[:, 2]

    @staticmethod
    def vx(solver):
        return solver.v[:, 0]

    @staticmethod
    def vy(solver):
        return solver.v[:, 1]

    @staticmethod
    def vz(solver):
        return solver.v[:, 2]

    def __del__(self):
        self.f.close()
