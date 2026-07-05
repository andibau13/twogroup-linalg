import cypari2
pari = cypari2.Pari()
import numpy as np

# converting between numpy and pari/flint/etc arrays
def pari_to_numpy(A):
    return np.array([[int(A[i, j]) for j in range(A.ncols())] for i in range(A.nrows())], dtype=int).reshape(A.nrows(),A.ncols())
def pari_to_numpy_mod8(A):
    return np.array([[int(A[i, j] % 8) for j in range(A.ncols())] for i in range(A.nrows())], dtype=int).reshape(A.nrows(),A.ncols())
def numpy_to_pari(A):
    return pari.matrix(A.shape[0], A.shape[1], [x for sub in A.tolist() for x in sub])

# wrapping smith normal form, hermite normal form, inverse, kernels, etc. into numpy arrays

# smith normal form
def snf(A):
    A_pari = numpy_to_pari(A)
    U_pari, V_pari, D_pari = A_pari.matsnf(1)
    U = pari_to_numpy(U_pari)
    V = pari_to_numpy(V_pari)
    D = pari_to_numpy(D_pari)
    return U, V, D

# smith normal form
def snf_mod8(A):
    A_pari = numpy_to_pari(A)
    U_pari, V_pari, D_pari = A_pari.matsnf(1)
    U = pari_to_numpy_mod8(U_pari)
    V = pari_to_numpy_mod8(V_pari)
    D = pari_to_numpy(D_pari)
    return U, V, D

# kernel of integer matrix
# this computes a LLL-reduced basis -> integers don't explode quite as quickly, but maybe that's the reason why it's slower
def int_kernel(A):
    A_pari = numpy_to_pari(A)
    ker_pari = A_pari.matkerint()
    ker = pari_to_numpy(ker_pari)
    return ker