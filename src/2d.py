from objectoriented import MolecularDynamics

obj = MolecularDynamics(numparticles=9, numdimensions=3, T=5, dt=0.001)
obj.initialize(position=[[0,0,0],[1.5,0,0],[0,1.5,0],[1.5,1.5,0],[-1.5,0,0],[-1.5,1.5,0],[0,-1.5,0],[1.5,-1.5,0],[-1.5,-1.5,0]])
obj.simulate(potential=obj.lennardJones, integrator=obj.velocityVerlet, dump=True)
