import kin_cython
import numpy as np

N = int(1E7)
v = np.random.normal(size=(N,3))

import time
start = time.clock()
kin_cython.K_cython(v)
end = time.clock()
print(end - start)
