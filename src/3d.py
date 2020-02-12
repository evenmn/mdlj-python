from objectoriented import MolecularDynamics

obj = MolecularDynamics(numparticles=500, numdimensions=3, T=10, dt=0.01)
obj.initialize(n=5, d=5)
obj.simulate(potential=obj.lennardJones, integrator=obj.velocityVerlet, dump=True)
obj.plot_energy()
