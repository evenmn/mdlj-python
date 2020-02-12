from objectoriented import MolecularDynamics

obj = MolecularDynamics(numparticles=2, numdimensions=3, T=5, dt=0.01)
obj.initialize(position=[[0,0,0],[1.5,0,0]])
obj.simulate(potential=obj.lennardJones, integrator=obj.eulerChromer)
obj.plot_distance()
