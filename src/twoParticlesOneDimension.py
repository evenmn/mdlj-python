from objectoriented import MolecularDynamics

# Two particles in one dimension separated by a distance 1.5 sigma
obj = MolecularDynamics(positions=[[0.0], [1.5]], T=5, dt=0.01)
obj.simulate(potential=obj.lennardJones, 
             integrator=obj.eulerChromer, 
             distance=True,
             poteng=True,
             dumpfile="../data/2N_1D_1.5S.data")
obj.plot_distance()
obj.plot_energy()

# Two particles in one dimension separated by a distance 0.95 sigma
obj = MolecularDynamics(positions=[[0.0], [0.95]], T=5, dt=0.01)
obj.simulate(potential=obj.lennardJones, 
             integrator=obj.eulerChromer, 
             distance=True,
             poteng=True,
             dumpfile="../data/2N_1D_0.95S.data")
obj.plot_distance()
obj.plot_energy()
