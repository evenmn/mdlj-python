import numpy as onp
import jax.numpy as np
import numba
from jax import jit

@numba.njit
def K_numba(v):
    K = 0
    N = len(v)
    for i in range(N):
        for j in range(3):
            K += v[i,j]**2
    return 0.5*K

def K_dum(v):
    K = 0
    N = len(v)
    for i in range(N):
        for j in range(3):
            K += v[i,j]**2
    return 0.5*K

def K_numpy(v):
    return 0.5*onp.sum(v**2)
    
def K_jax(v):
    return 0.5*np.sum(v**2)
    
@jit
def K_jax_jit(v):
    return 0.5*np.sum(v**2)
    

N = int(1E7)
v = np.random.normal(size=(N,3))

%timeit K_numba(v)
%timeit K_dum(v)
%timeit K_numpy(v)
%timeit K_jax(v)
%timeit K_jax_jit(v)
