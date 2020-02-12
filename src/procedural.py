import numpy as np
import matplotlib.pyplot as plt
import time

# constants
numparticles = 2
numdimensions = 3
T = 10
dt = 0.01
N = int(T/dt)

integrator = "eulerchromer"
potential = "lennardjones"

# declare arrays
time = np.linspace(0, T, N+1)
r = np.zeros((N+1, numparticles, numdimensions))
v = np.zeros((N+1, numparticles, numdimensions))
a = np.zeros((N+1, numparticles, numdimensions))
d = np.zeros((N+1, numparticles, numparticles))

# initialize arrays
r[0] = [[0,0,0],[1.5,0,0]]
d[0] = [[0, 1.5],[1.5,0]]
    
# integration loop
for t in range(N):

    # calculate distance matrix
    x = r[t][:,np.newaxis,:]
    y = r[t][np.newaxis,:,:]
    dr = x - y
    d[t+1] = np.linalg.norm(dr, axis=2)

    # calculate force acting on all the particles
    if potential == "lennardjones":
        k = np.square(d[t+1])
        l = np.nan_to_num(np.power(k, -3))
        m = np.square(l)
        n = np.reciprocal(k)
        factor = np.multiply(24 * (2 * m - l), n)
        factor[factor == np.inf] = 0
        a[t+1] = -np.einsum('ij,ijk->jk',factor,dr)
                
    else:
        raise NotImplementedError("Potential {} is not implemented.".format(potential))
            
    # integrate to find velocities and positions
    if integrator == "forwardeuler":
        v[t+1] = v[t] + a[t+1] * dt
        r[t+1] = r[t] + v[t] * dt
        
    elif integrator == "eulerchromer":
        v[t+1] = v[t] + a[t+1] * dt
        r[t+1] = r[t] + v[t+1] * dt
        
    elif integrator == "velocityverlet":
        v[t+1] = v[t] + 0.5 * (a[t+1] + a[t]) * dt
        r[t+1] = r[t] + v[t] * dt + 0.5 * a[t] * dt**2
        
    else:
        raise NotImplementedError("Integrator {} is not implemented.".format(integrator))

plt.plot(time, r[:,0,0])
plt.plot(time, r[:,1,0])
plt.show()
