# Python-MD
Molecular dynamics solver written in object-oriented Python. It is simple because it only supports symmetric systems, i.e., all sides have the same length and take the same boundary conditions. Furthermore, only the Lennard-Jones potential is implemented. Albeit efforts are put in making the code fast (mostly by replacing loops by vectorization operations), the performance cannot compete with packages written in low-level languages. 

## Set up solver
The first thing we need to do is to set up the solver. The solver is given by
``` python
MDSolver(positions, velocities, boundaries, T, dt)
```
where ```positions``` is the initialization of positions, ```velocities``` is the initialization of velocities, ```boundaries``` specifies the boundary conditions, ```T``` is the total simulation time and ```dt``` is the time step.

### Initialize position
One can initialize the positions in two different ways: manually by specifying the coordinate of every single particle or by choosing a face-centered cube. The initialization methods are found in the class ```InitPositions``` in ```initpositions.py```. 

#### Manual initialization
For manual initialization, call the class object ```SetPositions(positions)```. It takes an array_like object with the coordinates of all the particles. The number of particles and number of dimensions are set automatically.

**Example: Two particles in one dimension separated by a distance 1.5**
``` python
from moleculardynamics import MDSolver
from initpositions import SetPositions
solver = MDSolver(positions=SetPositions([[0.0], [1.5]]))
```

**Example: Four particles in two dimensions**
``` python
from moleculardynamics import MDSolver
from initpositions import SetPositions
solver = MDSolver(positions=SetPositions([[0,0], [0,2], [2,0], [2,2]]))
```

#### Face-centered cube
For face-centered cube, call the class object ```FCC(cells, lenbulk, dim)``` where ```cells``` is the number of cells in each direction, ```lenbulk``` is the length of the cube is each dimension and ```dim``` is the number of dimensions.

**Example: Four particles in three dimensions**
``` python
from moleculardynamics import MDSolver
from initpositions import FCC
solver = MDSolver(positions=FCC(cells=1, lenbulk=3, dim=3))
```

**Example: 864 particles in three dimensions**
``` python
from moleculardynamics import MDSolver
from initpositions import FCC
solver = MDSolver(positions=FCC(cells=6, lenbulk=10, dim=3))
```

### Initialize velocity
One can initialize the velocity in several different ways: manually, by a Gaussian distribution, by a given initial temperature and simply no initial velocity. The initialization methods are found in the class ```InitVelocities``` in ```initvelocities.py```.

#### No initial velocity
For no initial velocity, call the class object ```Zero()```. This is the default

**Example**
``` python
from moleculardynamics import MDSolver
from initvelocities import Zero
solver = MDSolver(velocities=Zero())
```

#### Manual initialization
For manual velocity initialization, call the class object ```SetVelocities(velocities)```, with ```velocities``` as a list of the velocities of all the particles.

**Example: Two particles in one dimension both moving with a velocity v=2.0**
``` python
from moleculardynamics import MDSolver
from initpositions import SetPositions
from initvelocities import SetVelocities
solver = MDSolver(positions=SetPositions([[0.0], [1.5]])
                  velocities=SetVelocities([[2.0], [2.0]]))
```

#### Temperature initialization
For initialization of the velocity according to some temperature, call the class object ```temperature(T)```. Here, ```T``` is the temperature given in Kelvin.

**Example: 864 particles with an initial temperature 300K**
``` python
from moleculardynamics import MDSolver
from initpositions import FCC
from initvelocities import Temperature
solver = MDSolver(positions=FCC(cells=6, lenbulk=10, dim=3),
                  velocities=Temperature(T=300))
```

### Boundary conditions
Three different boundary conditions are supported: open boundaries, reflective boundaries and periodic boundaries. The boundary methods are found in the class ```Boundaries``` in ```boundaryconditions.py```.

#### Open boundaries
For open boundaries, call the class object ```Open()```. This is the default.

**Example**
``` python
from moleculardynamics import MDSolver
from boundaryconditions import Open
solver = MDSolver(boundaries=Open())
```

#### Reflective boundaries
For reflective boundaries, call the class object ```Reflective(lenbox)```, with ```lenbox``` as the length of the box. 

**Example: 256 particles with reflective boundary conditions**
``` python
from moleculardynamics import MDSolver
from initpositions import FCC
from boundaryconditions import Reflective
solver = MDSolver(positions=FCC(cells=4, lenbulk=10,dim=3),
                  boundaries=Reflective(lenbox=10))
```

#### Periodic boundaries
For periodic boundaries, call the class object ```Periodic(lenbox)```, with ```lenbox``` as the length of the box. 

**Example: 256 particles with periodic boundary conditions**
``` python
from moleculardynamics import MDSolver
from initpositions import FCC
from boundaryconditions import Periodic
solver = MDSolver(positions=FCC(cells=4, lenbulk=10,dim=3),
                  boundaries=Periodic(lenbox=10))
```

### Time scale
The time scale is specified by ```T```, which is the total time and ```dt```, which is the time step.

**Example: 864 particles with an initial temperature 300K, periodic boundaries, simulated through T=5 ps with dt=0.01**
``` python
from moleculardynamics import MDSolver
from initpositions import FCC
from initvelocities import Temperature
from boundaryconditions import Periodic
solver = MDSolver(positions=FCC(cells=6, lenbulk=10, dim=3),
                  velocities=Temperature(T=300),
                  boundaries=Periodic(lenbox=10),
                  T=5,
                  dt=0.01)
```

## Simulate bulk
After the bulk is set up, we would like to see how it evolves in time. This is done by calling the function 
``` python
MDSolver.__call__(potential, integrator, poteng, distance, dumpfile)
```
where ```potential``` is a object specifying the inter-particle potential, ```integrator``` is a object specifying how to integrate the equation of motion, ```poteng``` is a boolean specifying whether or not the potential energy should be calculated, ```distance``` is a boolean specifying whether or not the distance matrix should be stored and ```dumpfile``` is a string specifying where to store the positions.

### Inter-particle potential
The inter-particle defines how the particles should interact. Potentials should be stored in the class ```Potential``` in ```potential.py``` Only the Lennard-Jones potential is implemented. 

#### Lennard-Jones
The lennard-Jones potential can by called by ```LennardJones(solver, cutoff)``` where ```solver``` is the solver object defined by the MDSolver and ```cutoff``` is the cutoff distance.

### Integrators
The integrators defines how to integrate the equation of motion, d^2r/dt^2=a. Integrators are stored in the class ```Integrator``` in ```integrator.py```. Implemented integrators are Forward-Euler, Euler-Chromer and Velocity-Verlet.

#### Forward-Euler
The forward-Euler integrator can by called by ```ForwardEuler(solver)``` where ```solver``` is the solver object defined by the MDSolver.

**Example: Simulate two particles on one dimension separated by a distance 1.5 using Lennard-Jones and forward-Euler**
``` python
from moleculardynamics import MDSolver
from initpositions import SetPositions
from potential import LennardJones
from integrator import ForwardEuler
solver = MDSolver(positions=SetPositions([[0.0], [1.5]])
                  T=5, dt=0.01)
solver(potential=LennardJones(solver, cutoff=3), 
       integrator=ForwardEuler(solver))
```

#### EulerChromer
The Euler-Chromer integrator can by called by ```EulerChromer(solver)``` where ```solver``` is the solver object defined by the MDSolver.

**Example: Simulate two particles on one dimension separated by a distance 1.5 using Lennard-Jones and Euler-Chromer**
``` python
from moleculardynamics import MDSolver
from initpositions import SetPositions
from potential import LennardJones
from integrator import EulerChromer
solver = MDSolver(positions=SetPositions([[0.0], [1.5]])
                  T=5, dt=0.01)
solver(potential=LennardJones(solver, cutoff=3), 
       integrator=EulerChromer(solver))
```

#### Velocity-Verlet
The VelocityVerlet integrator can by called by ```VelocityVerlet(solver)``` where ```solver``` is the solver object defined by the MDSolver.

**Example: Simulate two particles on one dimension separated by a distance 1.5 using Lennard-Jones and Velocity-Verlet**
``` python
from moleculardynamics import MDSolver
from initpositions import SetPositions
from potential import LennardJones
from integrator import VelocityVerlet
solver = MDSolver(positions=SetPositions([[0.0], [1.5]])
                  T=5, dt=0.01)
solver(potential=LennardJones(solver, cutoff=3), 
       integrator=VelocityVerlet(solver))
```

### Storage arguments
The remaining arguments are to specify what should be stored throughout the simulation. 

**Example: Simulate 864 particles with an initial temperature 300K, periodic boundaries, simulated through T=5 ps with dt=0.01 using LennardJones potential and Velocity-Verlet integrator***
``` python
from moleculardynamics import MDSolver
from initpositions import FCC
from initvelocities import Temperature
from boundaryconditions import Periodic
from potential import LennardJones
from integrator import VelocityVerlet
solver = MDSolver(positions=FCC(cells=6, lenbulk=10, dim=3),
                  velocities=Temperature(T=300),
                  boundaries=Periodic(lenbox=10),
                  T=5,
                  dt=0.01)
solver(potential=LennardJones(solver, cutoff=3), 
       integrator=VelocityVerlet(solver),
       distance=False,
       poteng=True,
       dumpfile="../data/864N_3D.data")
```

## Visualize
A few functions are implemented in order to plot the energy, distance and temperature. One can also easily visualize the particles using Ovito or VMD.

### Plot energy
To plot the energy, simply call

``` python
solver.plot_energy()
```
This requires that the potential energy is calculated during the simulation (```poteng=True```)

### Plot distance between particles
To plot the distance between particles, simply call

``` python
solver.plot_distance()
```
This requires that the distance matrix is stored throughout the simulation (```distance=True```)

NB: Not recommended for more than 4 particles, as the number of distances increases quadratically. 

### Plot temperature
To plot the temperature, simply call

``` python
solver.plot_temperature()
```

### Visualize the bulk using Ovito
The simulations can easily be visualized using Ovito by calling

``` bash
$ ovito dumpfile
```
where ```dumpfile``` is the xyz-file given in simulator.

## To do list
There are some issues with the code:

- Face-centered cube can be implemented more compact and general without the need of numpy 
- Make example directory
- Implement radial distribution function
