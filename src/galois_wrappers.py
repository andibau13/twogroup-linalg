import galois
GF2 = galois.GF(2)
import numpy as np

def kernel(A):
    return np.array(GF2(A % 2).null_space(), dtype=int).T
def rref(A):
    return np.array(GF2(A).row_reduce(), dtype=int)