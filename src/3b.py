from objectoriented import MolecularDynamics

obj = MolecularDynamics(numparticles=4, numdimensions=3, T=5, dt=0.0001)
obj.initialize(position=[[1,0.1,0],[0,1,0],[-1,0,0],[0,-1,0]])
obj.simulate(potential=obj.lennardJones, integrator=obj.velocityVerlet, dump=True)
obj.plot_energy()
