import numpy as np
from bisect import bisect_left
from .z2_helpers import *


# numpy dtype used to store hom.M and elem.v. Change to int32 or int64 if using Z_{2^i} for i>8
int_type = np.uint8

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

# helper function for get_item/set_item           
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

# helper function for get_item/set_item
def startstop2(dim0, dim1, key):
    if not (isinstance(key, tuple) and len(key)==2):
        raise ValueError("hom needs two indices")
    return startstop(dim0, key[0]) + startstop(dim1, key[1])

class Hom:
    """
    Represents a homomorphism between finite abelian 2-groups
    
    Attributes:
        M: np.array coefficient matrix
        dim0: "dimension" describing target 2-group. List of ints, such that the 2-group is Z2^dim0[0] x Z4^dim0[1] x ...
        dim1: dimension describing source 2-group.
    """

    def __init__(self, M, dim0, dim1):
        if M.shape[0] != sum(dim0):
            raise ValueError("Target dimension not matching (M.shape[0] != sum(dim0))")
        if M.shape[1] != sum(dim1):
            raise ValueError("source dimension not matching (M.shape[1] != sum(dim1))")
        self.M = np.asarray(M, dtype=int_type)
        self.dim0 = [int(i) for i in dim0]
        self.dim1 = [int(i) for i in dim1]

    def __getitem__(self, key):
        """
        X[i,j] accesses the coefficient block mapping between Z_{2^j} -> Z_{2^i}. Also implements simple slicing
        """
        start0, stop0, start1, stop1 = startstop2(self.dim0, self.dim1, key)
        return self.M[start0:stop0, start1:stop1]
    
    def __setitem__(self, key, value):
        """
        X[i,j] accesses the coefficient block mapping between Z_{2^j} -> Z_{2^i}. Also implements simple slicing
        """
        start0, stop0, start1, stop1 = startstop2(self.dim0, self.dim1, key)
        self.M[start0:stop0, start1:stop1] = value
        
    # generate random homomorphism between specific groups
    @staticmethod
    def rand(dim0, dim1):
        """
        Random homomorphism between two abelian 2-groups

        Args:
            dim0: Target 2-group
            dim1: Source 2-group

        Returns:
            random hom object between the prescribed 2-groups
        """
        rand = Hom(np.zeros((sum(dim0), sum(dim1)), dtype=int_type), dim0, dim1)
        for i in range(min(len(dim0), len(dim1))):
            rand[i:, i:] += (2**i) * np.random.randint(0,2,size=(sum(dim0[i:]),sum(dim1[i:])), dtype=int_type)
        return rand
    
    @staticmethod
    def rand_dim(max_dim, nr_dim0, nr_dim1):
        """
        Random homomorphism between random abelian 2-groups

        Args:
            max_dim: maximal number of copies of any Z_{2^i} factor
            nr_dim0: maximal i of a Z_{2^8} factor for target 2-group
            nr_dim1: maximal i of a Z_{2^8} factor for source 2-group
        """
        dim0 = np.random.randint(0,max_dim,size=(nr_dim0,))
        dim1 = np.random.randint(0,max_dim,size=(nr_dim1,))
        return Hom.rand(dim0, dim1)
    
    @staticmethod
    def rand_dim_nr(max_dim, max_nr_dim):
        nr_dim0 = np.random.randint(1,max_nr_dim+1)
        nr_dim1 = np.random.randint(1,max_nr_dim+1)
        return Hom.rand_dim(max_dim, nr_dim0, nr_dim1)
    
    @staticmethod
    def identity(dim):
        """
        Identity homomorphism

        Args:
            dim: 2-group on which the identity is returned
        """
        return Hom(np.eye(sum(dim), dtype=int_type), dim, dim)
    
    def zeros(dim0, dim1):
        """
        Zero homomorphism

        Args:
            dim0: target 2-group
            dim1: source 2-group
        """
        return Hom(np.zeros((sum(dim0), sum(dim1)), dtype=int_type), dim0, dim1)

    def reduce_mod(self):
        """
        Coefficients between Z_{2^i} and Z_{2^j} are valued in Z_{2^{min(i,j)}} but stored as uint8 integers. This function reduces the integers to the standard interval [0,...,2^{min(i,j)}-1]
        """
        for i in range(len(self.dim0)):
            for j in range(len(self.dim1)):
                self[i,j] %= 2**(min(i,j)+1)

    def tostring(self):
        """
        Prints hom object as string with horizontal and vertical line dividers between blocks of different i and j for the Z_{2^i} factors
        """
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

    def copy(self):
        """
        Deep copy
        """
        return Hom(self.M.copy(), self.dim0, self.dim1)
        
    def enhanced(self):
        """
        If the coefficient of a homomorphism between Z_{2^i} -> Z_{2^j} is "c", then the homomorphism acts by multiplication with c*2^{max(i,j)}. This function returns the matrix with coefficients c*2^{max(i,j)} instead.
        
        Returns:
            The output is a hom object but does **not** represent a homomorphism in the intended way.
        """
        M_enhance = self.copy()
        for i in range(len(M_enhance.dim0)):
            for j in range(len(M_enhance.dim1)):
                M_enhance[i,j] *= int(2**max(i-j, 0))
        return M_enhance

    def unenhanced(self):
        """
        Inverse of enhanced.
        """
        M_unenhance = self.copy()
        for i in range(len(M_unenhance.dim0)):
            for j in range(len(M_unenhance.dim1)):
                M_unenhance[i,j] //= int(2**max(i-j, 0))
        return M_unenhance
    

    def __matmul__(A, B):
        """
        Implements either (1) composition of homomorphisms, or (2) application of homomorphism to element

        Args:
            A: Hom object
            B: either (1) Hom object, or (2) Elem object

        Returns:
            Either (1) Composition AB, or (2) Application A(B)
        """
        if isinstance(B, Hom):
            check_dims_equal(A.dim1, B.dim0)
            AB = Hom(A.enhanced().M @ B.enhanced().M, A.dim0, B.dim1)
            AB = AB.unenhanced()
            AB.reduce_mod()
            return AB
        
        elif isinstance(B, Elem):
            check_dims_equal(A.dim1, B.dim)
            AB = Elem(A.enhanced().M @ B.v, A.dim0)
            AB.reduce_mod()
            return AB
        
        return NotImplemented
    
    def __add__(A, B):
        """
        Add two homomorphisms, (A+B)(x) = A(x) + B(x).
        """
        check_dims_equal(A.dim0, B.dim0)
        check_dims_equal(A.dim1, B.dim1)
        ApB = Hom(A.M+B.M, A.dim0, A.dim1)
        ApB.reduce_mod()
        return ApB
    
    def __sub__(A, B):
        """
        Subtract two homomorphisms
        """
        check_dims_equal(A.dim0, B.dim0)
        check_dims_equal(A.dim1, B.dim1)
        ApB = Hom(A.M-B.M, A.dim0, A.dim1)
        ApB.reduce_mod()
        return ApB
    
    def is_zero(self):
        """
        Test if homomorphism is zero
        """
        return np.all(self.M==0)
    
    def kernel(X, return_solve_helper = False):
        """
        Compute the kernel isomorphism of a homomorphism

        Args:
            X: homomorphism
            return_solve_helper: If True, also computes data that can be used to accelerate finding equations of the form Xa=b (see method solve_with_helper)

        Returns:
            K: kernel isomorphism: Injective homomorphism such that XK=0
            If return_solve_helper = True, also returns the helper
        """
        if return_solve_helper:
            Ks = []
            helps = []
        K = Hom.identity(X.dim1)
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
        
    def solve_with_helper(X, b, helper):
        """
        Computes a for k solution of the equation Xk=b

        Args:
            X: homomorphism
            b: element of target 2-group
            helper: this is an object returned from the kernel method if get_solve_helper = True is set.
                It is a pair (z2_helpers, K).
                z2_helpers is a list of length len(X.dim0). z2_helpers[i] stores helpers to solve the Z2 linear equation with the coefficient matrix for_L in the i-th iteration of the main loop of the kernel algorithm. The helpers contain the pivot column numbers and the RREF transform, see .z2_helpers.get_solve_helper and .z2_helpers.solve_with_helpers
                K is a list of length len(X.dim0). K[i] the kernel isomorphism K in the i-th iteration of the main loop of the kernel algorithm. K[0] is always the identity.

        Returns: Element k of source 2-group such that Xk=b, if exists

        Raises: ValueError if no solution exists
        """
        z2_helpers, K = helper

        for_l = b.v % 2
        k = Elem.zeros(X.dim1)
        for i in range(len(X.dim0)):
            try:
                l = Elem(solve_with_helper(*z2_helpers[i], for_l), K[i].dim1)
            except:
                raise ValueError("2-group linear equation has no solution.")
            k = k + K[i] @ l
            for_l = (b - X @ k).v // int(2**(i+1))

        return k
    
    def epi_mono(X):
        """
        Computes an epi-mono decomposition of the input homomorphism. that is, compute L, R, where L is surjective and R is injective, such that X=LR. L.dim1 == R.dim0 represents a 2-group that is isomorphic to both the image and cokernel of X

        Returns:
            homomorphisms L, R
        """
        n = X.dim0
        m = X.dim1

        L = Hom.zeros(n, [])
        R = Hom.zeros([], [])

        p_stack = [] # this is how Ri is embedded
        Y = np.zeros((sum(n), 0), dtype=int_type)
        multiple_Y = np.zeros((sum(n), 0), dtype=int_type)
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

            Ri = Hom(Ri_plus[:, multiple_Y.shape[1]:], list(reversed(L.dim1)), [m[i]] + nr_p_bars)

            p_bars = split_list(p_bar, Ri.dim1)
            nr_p_bars = [len(x) for x in p_bars]
            p_stack = [np.arange(m[i], dtype=int)] + p_stack

            R_new = Hom.zeros(L.dim1, m[i:])
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

        L.reduce_mod()
        R.reduce_mod()
        return L, R
    

    def transpose(self):
        """Transpose (dual) of a 2-group homomorphism.

        Under the perfect pairing <x, y> = sum_i x_i*y_i / 2^(l_i+1), every finite abelian 2-group is its own dual, and the dual of a homomorphism is given by the plain block-wise transpose of its coefficient matrix, with source and target 2-groups interchanged. (The stored block coefficients are unchanged since the enhancement factor 2^max(0, i-j) turns into 2^max(0, j-i) under dualization, and the value group Z_{2^(min(i,j)+1)} is symmetric.)
        """
        return Hom(self.M.T.copy(), self.dim1, self.dim0)

    def cokernel(self):
        """Cokernel projection of a homomorphism.

        Computed as the transpose of the kernel of the transpose (kernel and cokernel are exchanged under the self-duality of finite abelian 2-groups).

        Returns:
            The cokernel projection q: G -> G/im(self), a surjective Hom whose kernel is im(self)
        """
        return self.transpose().kernel().transpose()

    def solve_hom(K, P, K_solve_helper = None):
        """Solve K f = P for the homomorphism f: S -> T, given an injective K: T -> G and P: S -> G with im(P) contained in im(K).

        The solution is unique since K is injective. Works column-wise, solving K x = P(gen) for each generator gen of the source of P.

        Parameters:
            K: injective Hom T -> G
            P: Hom S -> G with im(P) contained in im(K)
            K_solve_helper: optional solve helper for K, as returned by K.kernel(return_solve_helper=True); computed on the fly if not given

        Returns:
            The unique Hom f: S -> T with K f = P
        """
        if K_solve_helper is None:
            _, K_solve_helper = K.kernel(return_solve_helper = True)

        # solve K f = P, one column (generator of the source of P) at a time
        f = Hom.zeros(K.dim1, P.dim1)
        for l in range(len(P.dim1)):
            for j in range(P.dim1[l]):
                gen = Elem.zeros(P.dim1)
                gen[l][j] = 1
                x = K.solve_with_helper(P @ gen, K_solve_helper)
                for i in range(len(K.dim1)):
                    if i <= l:
                        f[i, l][:, j] = x[i]
                    else:
                        # image of the generator at level i is c * 2^(i-l) with c the stored block coefficient;
                        # divisibility is guaranteed since 2^(l+1) * x = 0 by injectivity of K
                        assert np.all(x[i] % 2**(i-l) == 0)
                        f[i, l][:, j] = x[i] // 2**(i-l)

        return f

    def quotient_image_by_image(K, P, K_solve_helper = None):
        """Quotient the image of an injective homomorphism K: T -> G by the image of a homomorphism P: S -> G, given the promise im(P) is a subgroup of im(K).

        Combines solve_hom and cokernel: (1) solve K f = P for f: S -> T (unique since K is injective), then (2) take the cokernel of f.

        Parameters:
            K: injective Hom T -> G
            P: Hom S -> G with im(P) contained in im(K)
            K_solve_helper: optional solve helper for K, as returned by K.kernel(return_solve_helper=True); computed on the fly if not given

        Returns:
            The quotient projection q: T -> Q, a surjective Hom onto the quotient 2-group Q = im(K)/im(P), whose kernel is the preimage of im(P) under K
        """
        return K.solve_hom(P, K_solve_helper).cokernel()


class Elem:
    """
    Element of a 2-group

    Attributes:
        v: coefficient vector
        dim: 2-group of which elem is an element
    """

    def __init__(self, v, dim):
        if v.shape[0] != sum(dim):
            raise ValueError("Dimension of coefficient vector v does not match dim given")
        self.v = np.asarray(v, dtype=int_type)
        self.dim = dim

    def __getitem__(self, key):
        start, stop = startstop(self.dim, key)
        return self.v[start:stop]
    
    def __setitem__(self, key, value):
        start, stop = startstop(self.dim, key)
        self.v[start:stop] = value

    def reduce_mod(self):
        """
        Normalizes coefficients in Z_{2^i} block to the standard range [0,...,2^i-1]
        """
        for i in range(len(self.dim)):
            self[i] %= int(2**(i+1))

    def __add__(v,w):
        check_dims_equal(v.dim, w.dim)
        vpw = Elem(v.v + w.v, v.dim)
        vpw.reduce_mod()
        return vpw
    def __sub__(v,w):
        check_dims_equal(v.dim, w.dim)
        vmw = Elem(v.v - w.v, v.dim)
        vmw.reduce_mod()
        return vmw
    
    @staticmethod
    def rand(dim):
        """
        Random element of specified 2-group
        """
        tot_dim = sum(dim)
        rand = Elem(np.zeros((tot_dim,), dtype=int_type), dim)
        for i in range(len(dim)):
            rand[i] = np.random.randint(0, 2**(i+1), size = (dim[i],), dtype=int_type)
        return rand
    
    @staticmethod
    def zeros(dim):
        """
        Zero element of specified 2-group
        """
        tot_dim = sum(dim)
        return Elem(np.zeros((tot_dim,), dtype=int_type), dim)
    
    def is_zero(self):
        """
        Tests if element is zero
        """
        return np.all(self.v==0)
    
    def tostring(self):
        return " ".join([" ".join(self[j].astype(str).tolist()) + " |" for j in range(len(self.dim))])[:-2]

def to_z2_kernel(A: Hom, dim1):
    """
    Helper function for 2-group kernel isomorphism. Computes the kernel isomorphism of a homomorphism A from a 2-group (specified by dim1) to the group Z_2^i
    """
    Z, Z_dim = stagger_kernel(A % 2, dim1)
    Z_block = Hom(Z, dim1, Z_dim)
    W = []
    for i in range(1, len(dim1)):
        Zii = Z_block[i, i]
        W.append(image_completion(Zii))
    dim_tot = [Z_dim[i] + W[i].shape[1] for i in range(len(dim1)-1)] + [Z_dim[len(dim1)-1]]
    K = Hom.zeros(dim1, dim_tot)
    for i in range(0, len(dim1)):
        K[:, i][:, :Z_dim[i]] = Z_block[:, i]
    for i in range(0, len(dim1)-1):
        K[i+1, i][:, Z_dim[i]:] = W[i]
    return K


def split_list(values, lens):
    """
    Helper function for epi-mono decomposition (splits list of pivot column numbers)
    """
    result = []
    start = 0
    sep = 0
    for len in lens:
        sep += len
        end = bisect_left(values, sep, start)
        result.append([x-sep+len for x in values[start:end]])
        start = end
    return result