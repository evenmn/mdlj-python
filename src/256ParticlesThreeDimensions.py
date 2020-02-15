from objectoriented import MolecularDynamics

obj = MolecularDynamics(positions='fcc', cells=4, lencell=20, T=5, dt=0.01)
obj.simulate(potential=obj.lennardJones, integrator=obj.velocityVerlet, dumpfile="../data/256N_3D.data")
obj.plot_energy()
