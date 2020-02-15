from objectoriented import MolecularDynamics

obj = MolecularDynamics(positions='fcc', cells=5, lenbulk=10, T=5, dt=0.001)
obj.simulate(potential=obj.lennardJones, integrator=obj.velocityVerlet, dumpfile="../data/500N_3D.data")
obj.plot_energy()
