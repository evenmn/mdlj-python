from moleculardynamics import MDSolver

obj = MDSolver(positions='fcc', cells=5, lenbulk=10, T=5, dt=0.01)
obj.simulate(potential=obj.lennardJones, 
             integrator=obj.velocityVerlet, 
             poteng=True,
             dumpfile="../data/500N_3D.data")
obj.plot_energy()
