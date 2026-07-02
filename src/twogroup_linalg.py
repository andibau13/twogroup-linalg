import numpy as np
from bisect import bisect_left
from . import bitgauss_wrappers as z2lin
from .z2_helpers import *
# alternative below
# import galois_wrappers as z2lin

def check_dims_equal(dim0, dim1):
    mindim = min(len(dim0), len(dim1))
    for i in range(mindim):
        if dim0[i] != dim1[i]:
            raise ValueError(f"Z_(2^{i+1}) dimensions not matching")
        
    if len(dim0) != len(dim1):
        if len(dim0) > len(dim1):
            rem_dims = dim0[mindim:]
        if len(dim0) < len(dim1):
            rem_dims = dim1[mindim:]
        for i, dim in enumerate(rem_dims):
            if dim != 0:
                raise ValueError(f"Z_(2^{mindim + i+1}) dimensions not matching")
            
def startstop(dim, key):
    if isinstance(key, int):
        start0 = sum(dim[:key])
        stop0 = start0 + dim[key]
    if isinstance(key, slice):
        if key.start == None:
            start0 = 0
        else:
            start0 = sum(dim[:key.start])
        stop0 = start0 + sum(dim[key.start:key.stop])
    return start0, stop0

def startstop2(dim0, dim1, key):
    if not (isinstance(key, tuple) and len(key)==2):
        raise ValueError("hom needs two indices")
    return startstop(dim0, key[0]) + startstop(dim1, key[1])

# class storing homomorphisms between groups that are products of Z2, Z4, and Z8 factors
class hom:
    # M: matrix whose entries are the coefficients between pairs of individual Z2, Z4, and Z8 factors
    # dim0: triple (a,b,c) where a is the number of Z2 factors, b the number of Z4 factors, and c the number of Z8 factors of the output
    # dim1: same for the input of the homomorphism
    def __init__(self, M, dim0, dim1):
        if M.shape[0] != sum(dim0):
            raise ValueError("Dimension of axis 0 of matrix M does not match dim0 given")
        if M.shape[1] != sum(dim1):
            raise ValueError("Dimension of axis 1 of matrix M does not match dim1 given")
        self.M = M
        self.dim0 = dim0
        self.dim1 = dim1

    def __getitem__(self, key):
        start0, stop0, start1, stop1 = startstop2(self.dim0, self.dim1, key)
        return self.M[start0:stop0, start1:stop1]
    
    def __setitem__(self, key, value):
        start0, stop0, start1, stop1 = startstop2(self.dim0, self.dim1, key)
        self.M[start0:stop0, start1:stop1] = value
        
    # generate random homomorphism between specific groups
    @staticmethod
    def rand(dim0, dim1):
        rand = hom(np.zeros((sum(dim0), sum(dim1)),dtype=int), dim0, dim1)
        for i in range(min(len(dim0), len(dim1))):
            rand[i:, i:] += (2**i) * np.random.randint(0,2,size=(sum(dim0[i:]),sum(dim1[i:])))
        return rand
    
    @staticmethod
    def rand_dim(max_dim, nr0dim, nr1dim):
        dim0 = np.random.randint(0,max_dim,size=(nr0dim,))
        dim1 = np.random.randint(0,max_dim,size=(nr1dim,))
        return hom.rand(dim0, dim1)
    
    @staticmethod
    def identity(dim):
        return hom(np.eye(sum(dim), dtype=int), dim, dim)
    
    def zeros(dim0, dim1):
        return hom(np.zeros((sum(dim0), sum(dim1)), dtype=int), dim0, dim1)

    # the entries of M are defined either mod 2, mod 4, or mod 8. This function standardizes the entries to be between 0 and 2, 0 and 4, or 0 and 8, respectively.
    def reduce_mod(self):
        for i in range(len(self.dim0)):
            for j in range(len(self.dim1)):
                self[i,j] %= 2**(min(i,j)+1)

    # prints the matrix M defining the homomorphism to a string
    def tostring(self):
        out_string = ""
        col_widths = []
        for i, di in enumerate(self.dim1):
            inner_col_width = []
            for j in range(di):
                inner_col_width.append(max([1]+[len(str(x)) for x in self[:,i][:,j].tolist()]))
            col_widths.append(inner_col_width)
        row_separator = "".join(["-"*(sum(col_widths[i])+self.dim1[i]) + "+-" for i in range(len(self.dim1))])[:-2] + "\n"
        for i in range(len(self.dim0)):
            for d in range(self.dim0[i]):
                out_string += " ".join([" ".join([x.rjust(col_widths[j][y]) for y,x in enumerate(self[i,j][d,:].astype(str).tolist())] + ["|"]) for j in range(len(self.dim1))])[:-1] + "\n"
            out_string += row_separator
        return out_string[:-len(row_separator)]

    # provides a deep copy of M
    def copy(self):
        return hom(self.M.copy(), self.dim0, self.dim1)
        
    # multiplies submatrices of the M by factors of 2 and 4, such that it acts like integer matrix multiplication
    def enhanced(self):
        M_enhance = self.copy()
        for i in range(len(M_enhance.dim0)):
            for j in range(len(M_enhance.dim1)):
                M_enhance[i,j] *= int(2**max(i-j, 0))
        return M_enhance

    # inverse of enhanced
    def unenhanced(self):
        M_unenhance = self.copy()
        for i in range(len(M_unenhance.dim0)):
            for j in range(len(M_unenhance.dim1)):
                M_unenhance[i,j] //= int(2**max(i-j, 0))
        return M_unenhance
    

    # implements composition of homomorphisms, or application of homomorphism to element
    def __matmul__(A, B):
        if isinstance(B, hom):
            check_dims_equal(A.dim1, B.dim0)
            AB = hom(A.enhanced().M @ B.enhanced().M, A.dim0, B.dim1)
            AB = AB.unenhanced()
            AB.reduce_mod()
            return AB
        
        elif isinstance(B, elem):
            check_dims_equal(A.dim1, B.dim)
            AB = elem(A.enhanced().M @ B.v, A.dim0)
            AB.reduce_mod()
            return AB
        
        return NotImplemented
    
    def __add__(A, B):
        check_dims_equal(A.dim0, B.dim0)
        check_dims_equal(A.dim1, B.dim1)
        ApB = hom(A.M+B.M, A.dim0, A.dim1)
        ApB.reduce_mod()
        return ApB
    
    def __sub__(A, B):
        check_dims_equal(A.dim0, B.dim0)
        check_dims_equal(A.dim1, B.dim1)
        ApB = hom(A.M-B.M, A.dim0, A.dim1)
        ApB.reduce_mod()
        return ApB
    
    # remove all zero rows
    # def remove_zero_rows(self):
    #     non_zero_rows = np.any(self.M, axis=1)
    #     self.dim0 = (int(non_zero_rows[:self.dim0[0]].sum()), int(non_zero_rows[self.dim0[0]:self.dim0[0]+self.dim0[1]].sum()), int(non_zero_rows[self.dim0[0]+self.dim0[1]:].sum()))
    #     self.M = self.M[non_zero_rows,:]

    def is_zero(self):
        return np.all(self.M==0)
    
    # Find the kernel of a finite two-group homomorphism
    # Input A: z248_hom object
    def kernel(X, return_solve_helper = False):
        if return_solve_helper:
            Ks = []
            helps = []
        K = hom.identity(X.dim1)
        for i in range(len(X.dim0)):
            for_L = ((X @ K).enhanced().M // int(2**i)) % 2
            L = to_z2_kernel(for_L, K.dim1)
            if return_solve_helper:
                Ks.append(K)
                helps.append(get_solve_helper(for_L))
            K = K @ L
        if not return_solve_helper:
            return K
        else:
            return K, (helps, Ks)
        
    # find some arbitrary solution k to Xk=b
    # the helper is some data that is collected during the kernel computation for A
    # the first K is always the identity
    def solve_with_helper(X, b, helper):
        z2_helpers, K = helper

        for_l = b.v % 2
        k = elem.zeros(X.dim1)
        for i in range(len(X.dim0)):
            l = elem(solve_with_helper(*z2_helpers[i], for_l), K[i].dim1)
            k = k + K[i] @ l
            for_l = (b - X @ k).v // int(2**(i+1))

        # k0 = z248_elem(solve_with_helper(*z2_helpers[0], b.v % 2), X.dim1)
        # half_bmXk0 = (b - X @ k0).v // 2
        # l1 = z248_elem(solve_with_helper(*z2_helpers[1], half_bmXk0 % 2), K[0].dim1)
        # k1 = k0 + K[0] @ l1
        # quarter_bmXk1 = (b - X @ k1).v // 4
        # l2 = z248_elem(solve_with_helper(*z2_helpers[2], quarter_bmXk1 % 2), K[1].dim1)
        # k2 = k1 + K[1] @ l2
        return k
    

    # compute a surjective hom L and an injective hom R such that X=LR
    # L.dim1==R.dim0 represents the image of A as an abstract space, or equivalently the cokernel
    def epi_mono(X):
        n = X.dim0
        m = X.dim1

        L = hom.zeros(n, [])
        R = hom.zeros([], [])

        p_stack = [] # this is how Ri is embedded
        Y = np.zeros((sum(n), 0),dtype=int)
        multiple_Y = np.zeros((sum(n), 0),dtype=int)
        # update the current multiples of so-far generators
        # at every step attach current Y, then multiply by two. so after first step its two_y2
        nr_p_bars = []
        for i in reversed(range(len(m))):
            Y = np.hstack([X[:, i], Y])
            Z = np.hstack([multiple_Y, Y])
            Ri_plus, p_plus, p_bar_plus = rref_trim_pivs(Z[sum(n[:i]):, :] % 2)
            _, p = split_list(p_plus, [multiple_Y.shape[1], Y.shape[1]])
            _, p_bar = split_list(p_bar_plus, [multiple_Y.shape[1], Y.shape[1]])

            L.dim1 = [len(p)] + L.dim1
            L.M = np.hstack([Y[:, p], L.M])

            Ri = hom(Ri_plus[:, multiple_Y.shape[1]:], list(reversed(L.dim1)), [m[i]] + nr_p_bars)

            p_bars = split_list(p_bar, Ri.dim1)
            nr_p_bars = [len(x) for x in p_bars]
            p_stack = [np.arange(m[i], dtype=int)] + p_stack

            R_new = hom.zeros(L.dim1, m[i:])
            R_new[1:, 1:] += R.M
            for y in range(len(m)-i):
                for x in range(len(m)-i):
                    R_new[y, x][:, p_stack[x]] += 2**(min(x,y)) * Ri[len(m)-i-y-1, x]
            R = R_new

            p_stack = [mp[mpbar] for mpbar, mp in zip(p_bars, p_stack)]

            multiple_Y = np.hstack([multiple_Y, Y[:, p]])
            # multiply by 2 applies only to the first i blocks since the coefficient groups for the others change
            multiple_Y[:sum(n[:i])] *= 2

            Y = (Y - Z[:, p_plus] @ Ri.M)[:, p_bar]
            assert np.all(Y[sum(n[:i]):] % 2 == 0)
            Y[sum(n[:i]):, :] //= 2

        return L, R

class elem:
    # v: vector whose entries are the coefficients
    # dim: triple (a,b,c) where a is the number of Z2 factors, b the number of Z4 factors, and c the number of Z8 factors of the output
    def __init__(self, v, dim):
        if v.shape[0] != sum(dim):
            raise ValueError("Dimension of axis vector v does not match dim given")
        self.v = v
        self.dim = dim

    def __getitem__(self, key):
        start, stop = startstop(self.dim, key)
        return self.v[start:stop]
    
    def __setitem__(self, key, value):
        start, stop = startstop(self.dim, key)
        self.v[start:stop] = value

    def reduce_mod(self):
        for i in range(len(self.dim)):
            self[i] %= int(2**(i+1))

    def __add__(v,w):
        check_dims_equal(v.dim, w.dim)
        vpw = elem(v.v + w.v, v.dim)
        vpw.reduce_mod()
        return vpw
    def __sub__(v,w):
        check_dims_equal(v.dim, w.dim)
        vmw = elem(v.v - w.v, v.dim)
        vmw.reduce_mod()
        return vmw
    
    @staticmethod
    def rand(dim):
        tot_dim = sum(dim)
        rand = elem(np.zeros((tot_dim,), dtype=int), dim)
        for i in range(len(dim)):
            rand[i] = np.random.randint(0, 2**(i+1), size = (dim[i],))
        return rand
    
    @staticmethod
    def zeros(dim):
        tot_dim = sum(dim)
        return elem(np.zeros((tot_dim,), dtype=int), dim)
    
    def is_zero(self):
        return np.all(self.v==0)
    
    def tostring(self):
        return " ".join([" ".join(self[j].astype(str).tolist()) + " |" for j in range(len(self.dim))])[:-2]

# Find the kernel of a Z2 x Z4 x Z8 -> Z2 group homomorphism
# Input A: Integer numpy matrix (binary entries)
# Input dim1: tuple of the dimensions of the Z2, Z4, and Z8 part of A
# Returns: z248_hom object corresponding to the kernel isomorphism
def to_z2_kernel(A, dim1):
    Z, Z_dim = stagger_kernel(A % 2, dim1)
    Z_block = hom(Z, dim1, Z_dim)
    W = []
    for i in range(1, len(dim1)):
        Zii = Z_block[i, i]
        W.append(image_completion(Zii))
    dim_tot = [Z_dim[i] + W[i].shape[1] for i in range(len(dim1)-1)] + [Z_dim[len(dim1)-1]]
    K = hom.zeros(dim1, dim_tot)
    for i in range(0, len(dim1)):
        K[:, i][:, :Z_dim[i]] = Z_block[:, i]
    for i in range(0, len(dim1)-1):
        K[i+1, i][:, Z_dim[i]:] = W[i]
    return K




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

