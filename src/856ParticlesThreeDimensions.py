from objectoriented import MolecularDynamics

obj = MolecularDynamics(positions='fcc', cells=6, lenbulk=10, T=5, dt=0.001)
obj.simulate(potential=obj.lennardJones, 
             integrator=obj.velocityVerlet,
             poteng=True,
             dumpfile="../data/856N_3D.data")
obj.plot_energy()
