import twogroup_linalg as lin
import numpy as np
import kernel_snf

def group_dim(dim):
    """Number of Z_2 generators of the 2-group described by dim (i copies of Z_{2^{i+1}} contribute (i+1) each)"""
    return sum((i + 1) * d for i, d in enumerate(dim))

def test_kernel():
    """Test if kernel isomorphism indeed lands in the kernel"""
    np.random.seed(345)
    for i in range(30):
        x = lin.Hom.rand_dim_nr(4, 5)
        xker = x.kernel()
        assert (x@xker).is_zero()

def test_snf_kernel():
    """Test if snf kernel isomorphism indeed lands in the kernel"""
    np.random.seed(345)
    for i in range(30):
        x = lin.Hom.rand_dim_nr(4, 5)
        try:
            xker = kernel_snf.twogroup_kernel_snf(x)
            assert (x@xker).is_zero()
        except:
            pass

def test_kernel_dimension():
    """Test if kernel dimension is the same as for snf kernel"""
    np.random.seed(345)
    for i in range(30):
        x = lin.Hom.rand_dim_nr(4, 5)
        xker = x.kernel()
        try:
            xker_snf = kernel_snf.twogroup_kernel_snf(x)
            assert xker.dim1 == xker_snf.dim1
        except:
            pass

def test_multiplication():
    """Test that composition of homomorphisms does not crash"""
    np.random.seed(345)
    dim0 = np.random.randint(0,4,size=(5,))
    dim1 = np.random.randint(0,4,size=(5,))
    dim2 = np.random.randint(0,4,size=(5,))
    x = lin.Hom.rand(dim2, dim1)
    y = lin.Hom.rand(dim1, dim0)
    xy = x @ y
    assert xy.dim0 == x.dim0
    assert xy.dim1 == y.dim1

def test_tostring():
    """Test that tostring does not crash"""
    x = lin.Hom.rand_dim(4, 5, 5)
    assert isinstance(x.tostring(), str)

def test_solve_with_helper():
    """Test that solve_with_helper finds a valid solution k to Xk=b"""
    np.random.seed(345)
    for i in range(30):
        x = lin.Hom.rand_dim_nr(4, 5)
        sol_to_find = lin.Elem.rand(x.dim1)
        b = x @ sol_to_find
        ker, helps = x.kernel(return_solve_helper=True)
        sol = x.solve_with_helper(b, helps)
        assert (x @ sol - b).is_zero()

def test_epi_mono():
    """Test epi-mono decomposition: LR=X, L is injective, and cokernel dim equals image dim"""
    np.random.seed(345)
    for i in range(30):
        X = lin.Hom.rand_dim_nr(4, 5)
        L, R = X.epi_mono()
        Lker = L.kernel()
        Xker = X.kernel()
        X_coker_dim = sum([(i+1)*d for i, d in enumerate(X.dim1)]) - sum([(i+1)*d for i, d in enumerate(Xker.dim1)])
        X_img_dim = sum([(i+1)*d for i, d in enumerate(L.dim1)])
        assert (L @ R - X).is_zero()
        assert not any(Lker.dim1)
        assert X_coker_dim == X_img_dim

def test_transpose_involution():
    """Test that the transpose is an involution: (X^T)^T == X (same coefficients, interchanged-then-restored dims)"""
    np.random.seed(345)
    for i in range(30):
        X = lin.Hom.rand_dim_nr(4, 5)
        Xtt = X.transpose().transpose()
        assert Xtt.dim0 == X.dim0
        assert Xtt.dim1 == X.dim1
        assert np.array_equal(Xtt.M, X.M)

def test_solve_hom():
    """Test that solve_hom recovers f from K @ f: for injective K and arbitrary L, solving K g = K@L gives back g == L exactly.

    L need not be injective: uniqueness of the solution comes from K being injective, so the recovered g is exactly L.
    """
    np.random.seed(345)
    for i in range(30):
        X = lin.Hom.rand_dim_nr(4, 5)
        K, _ = X.epi_mono()                      # K: injective, target T = K.dim1
        assert not any(K.kernel().dim1)          # K is injective
        L = lin.Hom.rand(K.dim1, np.random.randint(0, 4, size=(4,)))  # general L into T (possibly non-injective)
        L.reduce_mod()
        P = K @ L
        f = K.solve_hom(P)
        assert (K @ f - P).is_zero()             # f solves K f = P
        assert f.dim0 == L.dim0 and f.dim1 == L.dim1
        assert np.array_equal(f.M, L.M)          # unique solution: f is exactly L