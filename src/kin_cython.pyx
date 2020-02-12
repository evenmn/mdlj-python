def K_cython(v):
    K = 0
    N = len(v)
    for i in range(N):
        for j in range(3):
            K += v[i,j]**2
    return 0.5*K
