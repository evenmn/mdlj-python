from objectoriented import MolecularDynamics

obj = MolecularDynamics(positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], T=5, dt=0.01)
obj.simulate(potential=obj.lennardJones, integrator=obj.eulerChromer, dumpfile="../data/2b1.data")
obj.plot_distance()

#obj.initialize(position=[[0,0,0],[0.95,0,0]])
#obj.simulate(potential=obj.lennardJones, integrator=obj.eulerChromer, dumpfile="../data/2b2.data")
#obj.plot_distance()
