import numpy as np
import matplotlib.pyplot as plt
plt.style.use("bmh")

def lennard_jones(r, sigma, epsilon):
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
    
r = np.linspace(0.9, 3, 1000)

plt.plot(r, lennard_jones(r, 1, 1))
plt.xlabel("Distance $[r/\sigma$]")
plt.ylabel(r"Energy, [$U(r)/\varepsilon$]")
plt.show()
