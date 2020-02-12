from objectoriented import MolecularDynamics

obj = MolecularDynamics(numparticles=108, numdimensions=3, T=5, dt=0.001)
obj.initialize(n=3, d=20)
obj.simulate(potential=obj.lennardJones, integrator=obj.velocityVerlet)
obj.plot_energy()
