import numpy as np
from bisect import bisect_left
from . import bitgauss_wrappers as z2lin
from .z2_helpers import *

class mod_matrix:
    """
    Matrix A of numbers mod 2^i, stored as a list of binary matrixes A_0, A_1, ..., A_(i-1), such that A = A_0+2A_1+4A_2+...
    Attributes:
        matrices: List of binary matrices, stored as bitgauss BitMatrix
        level: The integer i
        shape: The shape of the matrix
    """
    def __init__(self, shape: tuple[int,int], matrices: list[z2lin.BitMatrix], level: int):
        self.shape = shape
        self.matrices = matrices
        self.level = level
        for m in matrices:
            if (m.cols, m.rows) != shape:
                raise ValueError("shape of Bitmatrices needs to match given shape")

    def __add__(A, B):
        if A.shape != B.shape:
            raise ValueError("shape of mod_matrix objects needs to match")
        if A.level != B.level:
            raise ValueError("level of mod_matrix objects needs to match")

        carry = z2lin.BitMatrix.from_numpy_bool(np.zeros(A.shape, dtype=bool))
        matrices = []
        for i in range(A.level):
            Ai_p_Bi = A.matrices[i] + B.matrices[i]
            matrices.append(Ai_p_Bi + carry)
            carry = (A.matrices[i] & B.matrices[i]) + (carry & Ai_p_Bi)

        return mod_matrix(A.shape, matrices, A.level)

    def product(A, B, output_level):
        if A.shape[1] != B.shape[0]:
            raise ValueError("inner dimensions of mod_matrix objects need to match")

        output_shape = (A.shape[0], B.shape[1])

        def zero_matrix():
            return z2lin.BitMatrix.from_numpy_bool(np.zeros(output_shape, dtype=bool))

        result = mod_matrix(output_shape, [zero_matrix() for _ in range(output_level)], output_level)
        for i in range(min(A.level, output_level)):
            for j in range(min(B.level, output_level - i)):
                shift = i + j
                partial = mul_mod(A.matrices[i], B.matrices[j], output_level - shift)
                shifted_matrices = [zero_matrix() for _ in range(shift)] + partial.matrices
                result = result + mod_matrix(output_shape, shifted_matrices, output_level)

        return result

    def to_numpy(self):
        M = np.zeros(self.shape, dtype=int)
        for i in range(self.level):
            M += (2**i) * z2lin.bitgauss_to_numpy(self.matrices[i])
        return M

    @staticmethod
    def from_numpy(M, level):
        M = np.asarray(M, dtype=int)
        matrices = []
        for i in range(level):
            matrices.append(z2lin.numpy_to_bitgauss((M // (2**i)) % 2))
        return mod_matrix(M.shape, matrices, level)
    
def mul_mod(A: z2lin.BitMatrix, B: z2lin.BitMatrix, level: int):
    matrices = z2lin.BitMatrix.mul_carries(A, B, level)
    return mod_matrix((A.cols, B.rows), matrices, level)