from moleculardynamics import MDSolver

obj = MDSolver(positions=[[0, 0], [2, 0], [0, 2], [1.5, 1.5]],
               T=5, 
               dt=0.01)
obj.simulate(potential=obj.lennardJones, 
             integrator=obj.eulerChromer, 
             distance=True,
             poteng=True,
             dumpfile="../data/4N_2D.data")
obj.plot_distance()
obj.plot_energy()
