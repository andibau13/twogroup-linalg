import numpy as np
from bisect import bisect_left
from . import bitgauss_wrappers as z2lin
from .z2_helpers import *
# alternative below
# import galois_wrappers as z2lin

def check_dims_equal(dim0, dim1):
    if dim1[0] != dim0[0]:
        raise ValueError("Z2 dimensions not matching")
    if dim1[1] != dim0[1]:
        raise ValueError("Z4 dimensions not matching")
    if dim1[2] != dim0[2]:
        raise ValueError("Z8 dimensions not matching")

# class storing homomorphisms between groups that are products of Z2, Z4, and Z8 factors
class z248_hom:
    # M: matrix whose entries are the coefficients between pairs of individual Z2, Z4, and Z8 factors
    # dim0: triple (a,b,c) where a is the number of Z2 factors, b the number of Z4 factors, and c the number of Z8 factors of the output
    # dim1: same for the input of the homomorphism
    def __init__(self, M, dim0, dim1):
        if M.shape[0] != dim0[0]+dim0[1]+dim0[2]:
            raise ValueError("Dimension of axis 0 of matrix M does not match dim0 given")
        if M.shape[1] != dim1[0]+dim1[1]+dim1[2]:
            raise ValueError("Dimension of axis 1 of matrix M does not match dim1 given")
        self.M = M
        self.dim0 = dim0
        self.dim1 = dim1

    # generate random homomorphism between specific groups
    @staticmethod
    def rand(dim0, dim1):
        rand = np.random.randint(0,2,size=(dim0[0]+dim0[1]+dim0[2],dim1[0]+dim1[1]+dim1[2]))
        rand[dim0[0]:,dim1[0]:] += 2* np.random.randint(0,2,size=(dim0[1]+dim0[2],dim1[1]+dim1[2]))
        rand[dim0[0]+dim0[1]:,dim1[0]+dim1[1]:] += 4* np.random.randint(0,2,size=(dim0[2],dim1[2]))
        return z248_hom(rand, dim0, dim1)
    
    @staticmethod
    def rand_dim(max_dim):
        dim0 = np.random.randint(0,max_dim,size=(3,))
        dim1 = np.random.randint(0,max_dim,size=(3,))
        return z248_hom.rand(dim0, dim1)

    # the entries of M are defined either mod 2, mod 4, or mod 8. This function standardizes the entries to be between 0 and 2, 0 and 4, or 0 and 8, respectively.
    def reduce_mod(self):
        M_mod = self.M % 2
        M_mod[self.dim0[0]:,self.dim1[0]:] = self.M[self.dim0[0]:,self.dim1[0]:] % 4
        M_mod[self.dim0[0]+self.dim0[1]:, self.dim1[0]+self.dim1[1]:] = self.M[self.dim0[0]+self.dim0[1]:, self.dim1[0]+self.dim1[1]:] % 8
        self.M = M_mod

    # prints the matrix M defining the homomorphism to a string
    def tostring(self):
        def part_string(X):
            X_string = ""
            for x in X:
                X_string += " ".join(x[:self.dim1[0]].astype(str).tolist() + ["|"] + x[self.dim1[0]:self.dim1[0]+self.dim1[1]].astype(str).tolist() + ["|"] + x[self.dim1[0]+self.dim1[1]:].astype(str).tolist() + ["\n"])
            return X_string
        row_separator = "--"*self.dim1[0] + "+-" + "--"*self.dim1[1] + "+-" + "--"*self.dim1[2] + "\n"
        return part_string(self.M[:self.dim0[0]]) + row_separator + part_string(self.M[self.dim0[0]:self.dim0[0]+self.dim0[1]]) + row_separator + part_string(self.M[self.dim0[0]+self.dim0[1]:])

    # provides a deep copy of M
    def copy(self):
        return z248_hom(self.M.copy(), self.dim0, self.dim1)
        
    # multiplies submatrices of the M by factors of 2 and 4, such that it acts like integer matrix multiplication
    def enhanced(self):
        M_enhance = self.M.copy()
        M_enhance[self.dim0[0]:self.dim0[0]+self.dim0[1],:self.dim1[0]] = 2* self.M[self.dim0[0]:self.dim0[0]+self.dim0[1],:self.dim1[0]]
        M_enhance[self.dim0[0]+self.dim0[1]:, self.dim1[0]:self.dim1[0]+self.dim1[1]] = 2*self.M[self.dim0[0]+self.dim0[1]:, self.dim1[0]:self.dim1[0]+self.dim1[1]]
        M_enhance[self.dim0[0]+self.dim0[1]:,:self.dim1[0]] = 4*self.M[self.dim0[0]+self.dim0[1]:,:self.dim1[0]]
        return M_enhance

    # inverse of enhanced
    def unenhanced(self):
        M_unenhance = self.M.copy()
        M_unenhance[self.dim0[0]:self.dim0[0]+self.dim0[1],:self.dim1[0]] = self.M[self.dim0[0]:self.dim0[0]+self.dim0[1],:self.dim1[0]] // 2
        M_unenhance[self.dim0[0]+self.dim0[1]:, self.dim1[0]:self.dim1[0]+self.dim1[1]] = self.M[self.dim0[0]+self.dim0[1]:, self.dim1[0]:self.dim1[0]+self.dim1[1]] // 2
        M_unenhance[self.dim0[0]+self.dim0[1]:,:self.dim1[0]] = self.M[self.dim0[0]+self.dim0[1]:,:self.dim1[0]] // 4
        return M_unenhance

    # implements composition of homomorphisms, or application of homomorphism to element
    def __matmul__(A, B):
        if isinstance(B, z248_hom):
            check_dims_equal(A.dim1, B.dim0)
            AB = z248_hom(A.enhanced() @ B.enhanced(), A.dim0, B.dim1)
            AB.M = AB.unenhanced()
            AB.reduce_mod()
            return AB
        
        elif isinstance(B, z248_elem):
            check_dims_equal(A.dim1, B.dim)
            AB = z248_elem(A.enhanced() @ B.v, A.dim0)
            AB.reduce_mod()
            return AB
        
        return NotImplemented
    
    def __add__(A, B):
        check_dims_equal(A.dim0, B.dim0)
        check_dims_equal(A.dim1, B.dim1)
        ApB = z248_hom(A.M+B.M, A.dim0, A.dim1)
        ApB.reduce_mod()
        return ApB
    
    def __sub__(A, B):
        check_dims_equal(A.dim0, B.dim0)
        check_dims_equal(A.dim1, B.dim1)
        ApB = z248_hom(A.M-B.M, A.dim0, A.dim1)
        ApB.reduce_mod()
        return ApB
    
    # remove all zero rows
    def remove_zero_rows(self):
        non_zero_rows = np.any(self.M, axis=1)
        self.dim0 = (int(non_zero_rows[:self.dim0[0]].sum()), int(non_zero_rows[self.dim0[0]:self.dim0[0]+self.dim0[1]].sum()), int(non_zero_rows[self.dim0[0]+self.dim0[1]:].sum()))
        self.M = self.M[non_zero_rows,:]

    def is_zero(self):
        return np.all(self.M==0)

class z248_elem:
    # v: vector whose entries are the coefficients
    # dim: triple (a,b,c) where a is the number of Z2 factors, b the number of Z4 factors, and c the number of Z8 factors of the output
    def __init__(self, v, dim):
        if v.shape[0] != dim[0]+dim[1]+dim[2]:
            raise ValueError("Dimension of axis vector v does not match dim given")
        self.v = v
        self.dim = dim

    def reduce_mod(self):
        self.v[:self.dim[0]] %= 2
        self.v[self.dim[0]:self.dim[0]+self.dim[1]] %= 4
        self.v[self.dim[0]+self.dim[1]:] %= 8

    def __add__(v,w):
        check_dims_equal(v.dim, w.dim)
        vpw = z248_elem(v.v + w.v, v.dim)
        vpw.reduce_mod()
        return vpw
    def __sub__(v,w):
        check_dims_equal(v.dim, w.dim)
        vmw = z248_elem(v.v - w.v, v.dim)
        vmw.reduce_mod()
        return vmw
    
    @staticmethod
    def rand(dim):
        tot_dim = dim[0]+dim[1]+dim[2]
        rand = z248_elem(np.random.randint(0,8,size=(tot_dim,)), dim)
        rand.reduce_mod()
        return rand
    
    def is_zero(self):
        return np.all(self.v==0)
    
    def tostring(self):
        elem_string = ""
        for x in self.v[:self.dim[0]]:
            elem_string += str(x) + " "
        elem_string += "| "
        for x in self.v[self.dim[0]:self.dim[0]+self.dim[1]]:
            elem_string += str(x) + " "
        elem_string += "| "
        for x in self.v[self.dim[0]+self.dim[1]:]:
            elem_string += str(x) + " "
        return elem_string
        

# Find the kernel of a Z2 x Z4 x Z8 -> Z2 group homomorphism
# Input A: Integer numpy matrix (binary entries)
# Input dim1: tuple of the dimensions of the Z2, Z4, and Z8 part of A
# Returns: z248_hom object corresponding to the kernel isomorphism
def z248_z2_kernel(A, dim1):
    Z, Z_dim = stagger_kernel(A % 2, dim1)
    Z22 = Z[dim1[0]+dim1[1]:, Z_dim[0]+Z_dim[1]:]
    W2 = image_completion(Z22)
    Z11 = Z[dim1[0]:dim1[0]+dim1[1], Z_dim[0]:Z_dim[0]+Z_dim[1]]
    W1 = image_completion(Z11)

    Z00_zero_zero = Z[:, :Z_dim[0]]
    Z01_Z11_zero = Z[:, Z_dim[0]:Z_dim[0]+Z_dim[1]]
    Z02_Z12_Z22 = Z[:, Z_dim[0]+Z_dim[1]:]
    zero_W1_zero = np.vstack([np.zeros((dim1[0], W1.shape[1]), dtype=int), W1, np.zeros((dim1[2], W1.shape[1]), dtype=int)])
    zero_zero_W2 = np.vstack([np.zeros((dim1[0]+dim1[1], W2.shape[1]), dtype=int), W2])
    
    K = np.hstack([Z00_zero_zero, zero_W1_zero, Z01_Z11_zero, zero_zero_W2, Z02_Z12_Z22])
    dim2 = (Z_dim[0] + W1.shape[1], Z_dim[1] + W2.shape[1], Z_dim[2])
    return z248_hom(K, dim1, dim2)

# Find the kernel of a Z2 x Z4 x Z8 -> Z2 x Z4 x Z8 group homomorphism
# Input A: z248_hom object
def z248_z248_kernel(X, return_solve_helper = False):
    m2_X = X.enhanced() % 2
    K0 = z248_z2_kernel(m2_X, X.dim1)
    half_X_K0 = ((X @ K0).enhanced() // 2) % 2
    L1 = z248_z2_kernel(half_X_K0, K0.dim1)
    K1 = K0 @ L1
    quarter_X_K1 = ((X @ K1).enhanced() // 4) % 2
    L2 = z248_z2_kernel(quarter_X_K1, K1.dim1)
    K2 = K1 @ L2
    if not return_solve_helper:
        return K2
    else:
        return K2, ([get_solve_helper(m2_X), get_solve_helper(half_X_K0), get_solve_helper(quarter_X_K1)], [K0, K1])

z248_hom.kernel = z248_z248_kernel

# find some arbitrary solution to Ax=b
# the helper is some data that is collected during the kernel computation for A
def z248_solve_with_helper(X, b, helper):
    z2_helpers, K = helper
    k0 = z248_elem(solve_with_helper(*z2_helpers[0], b.v % 2), X.dim1)
    half_bmXk0 = (b - X @ k0).v // 2
    l1 = z248_elem(solve_with_helper(*z2_helpers[1], half_bmXk0 % 2), K[0].dim1)
    k1 = k0 + K[0] @ l1
    quarter_bmXk1 = (b - X @ k1).v // 4
    l2 = z248_elem(solve_with_helper(*z2_helpers[2], quarter_bmXk1 % 2), K[1].dim1)
    k2 = k1 + K[1] @ l2
    return k2


def split_list(values, lens):
    result = []
    start = 0
    sep = 0
    for len in lens:
        sep += len
        end = bisect_left(values, sep, start)
        result.append([x-sep+len for x in values[start:end]])
        start = end
    return result


# compute a surjective hom L and an injective hom R such that X=LR
# L.dim1==R.dim0 represents the image of A as an abstract space, or equivalently the cokernel
def z248_epi_mono(hom):
    n = hom.dim0
    m = hom.dim1
    X = hom.M

    Y2 = X[:, m[0]+m[1]:]
    Z2 = Y2
    R2, p2, p2_bar = rref_trim_pivs(Z2[n[0]+n[1]:, :] % 2)
    p22_bar = p2_bar

    Y1 = (Y2 - Z2[:, p2] @ R2)[:, p2_bar]
    Y1[n[0]+n[1]:, :] //= 2
    Y1 = np.hstack([X[:, m[0]:m[0]+m[1]], Y1])
    two_Y2 = np.vstack([np.zeros((n[0], len(p2)),dtype=int), 2*Y2[n[0]:n[0]+n[1], p2], Y2[n[0]+n[1]:, p2]])
    Z1 = np.hstack([two_Y2, Y1])
    R1, p1_plus, p1_bar_plus = rref_trim_pivs(Z1[n[0]:, :] % 2)
    R1 = R1[:, len(p2):]
    _, p1 = split_list(p1_plus, [len(p2), m[1] + len(p22_bar)])
    _, p1_bar = split_list(p1_bar_plus, [len(p2), m[1] + len(p22_bar)])
    p11_bar, p12_bar = split_list(p1_bar, [m[1], len(p22_bar)])

    Y0 = (Y1 - Z1[:, p1_plus] @ R1)[:, p1_bar]
    Y0[n[0]:, :] //= 2
    Y0 = np.hstack([X[:, :m[0]], Y0])
    four_Y2 = np.vstack([np.zeros((n[0]+n[1], len(p2)),dtype=int), Y2[n[0]+n[1]:, p2]])
    two_Y1 = np.vstack([np.zeros((n[0],len(p1)),dtype=int), Y1[n[0]:, p1]])
    Z0 = np.hstack([four_Y2, two_Y1, Y0])
    R0, p0_plus, p0_bar_plus = rref_trim_pivs(Z0 % 2)
    R0 = R0[:, len(p2)+len(p1):]
    _, p0 = split_list(p0_plus, [len(p2)+len(p1), m[0]+len(p11_bar)+len(p12_bar)])
    _, p0_bar = split_list(p0_bar_plus, [len(p2)+len(p1), m[0]+len(p11_bar)+len(p12_bar)])
    p00_bar, p01_bar, p02_bar = split_list(p0_bar, [m[0], len(p11_bar), len(p12_bar)])

    L = np.hstack([Y0[:, p0], Y1[:, p1], Y2[:, p2]])

    R = np.zeros((len(p0)+len(p1)+len(p2), m[0]+m[1]+m[2]), dtype=int)
    R[:len(p0), :m[0]] += R0[len(p2)+len(p1):, :m[0]]
    R[len(p0):len(p0)+len(p1), :m[0]] += R0[len(p2):len(p2)+len(p1), :m[0]]
    R[len(p0)+len(p1):, :m[0]] += R0[:len(p2), :m[0]]

    R[:len(p0), m[0]:m[0]+m[1]][:, p11_bar] += R0[len(p2)+len(p1):, m[0]:m[0]+len(p11_bar)]
    R[len(p0):len(p0)+len(p1), m[0]:m[0]+m[1]] += R1[len(p2):len(p2)+len(p1), :m[1]]
    R[len(p0):len(p0)+len(p1), m[0]:m[0]+m[1]][:, p11_bar] += 2* R0[len(p2):len(p2)+len(p1), m[0]:m[0]+len(p11_bar)]
    R[len(p0)+len(p1):, m[0]:m[0]+m[1]] += R1[:len(p2), :m[1]]
    R[len(p0)+len(p1):, m[0]:m[0]+m[1]][:, p11_bar] += 2* R0[:len(p2), m[0]:m[0]+len(p11_bar)]
    
    R[:len(p0), m[0]+m[1]:][:, np.array(p22_bar,dtype=int)[p12_bar]] += R0[len(p2)+len(p1):, m[0]+len(p11_bar):]
    R[len(p0):len(p0)+len(p1), m[0]+m[1]:][:, p22_bar] += R1[len(p2):, m[1]:]
    R[len(p0):len(p0)+len(p1), m[0]+m[1]:][:, np.array(p22_bar,dtype=int)[p12_bar]] += 2* R0[len(p2):len(p2)+len(p1), m[0]+len(p11_bar):]
    R[len(p0)+len(p1):, m[0]+m[1]:] += R2
    R[len(p0)+len(p1):, m[0]+m[1]:][:, p22_bar] += 2* R1[:len(p2), m[1]:]
    R[len(p0)+len(p1):, m[0]+m[1]:][:, np.array(p22_bar,dtype=int)[p12_bar]] += 4* R0[:len(p2), m[0]+len(p11_bar):]

    img_dim = (len(p0), len(p1), len(p2))
    L_hom = z248_hom(L, n, img_dim)
    R_hom = z248_hom(R, img_dim, m)
    L_hom.reduce_mod()
    R_hom.reduce_mod()

    return L_hom, R_hom

