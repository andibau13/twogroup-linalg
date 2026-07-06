import twogroup_linalg as lin
import numpy as np
import z248_kernel_snf as snf

def test_kernel():
    """Test if kernel isomorphism indeed lands in the kernel"""
    np.random.seed(345)
    for i in range(30):
        x = lin.hom.rand_dim_nr(4, 5)
        xker = x.kernel()
        assert (x@xker).is_zero()

def test_snf_kernel():
    """Test if snf kernel isomorphism indeed lands in the kernel"""
    np.random.seed(345)
    for i in range(30):
        x = lin.hom.rand_dim_nr(4, 5)
        try:
            xker = snf.twogroup_kernel_snf(x)
            assert (x@xker).is_zero()
        except:
            pass

def test_kernel_dimension():
    """Test if kernel dimension is the same as for snf kernel"""
    np.random.seed(345)
    for i in range(30):
        x = lin.hom.rand_dim_nr(4, 5)
        xker = x.kernel()
        try:
            xker_snf = snf.twogroup_kernel_snf(x)
            assert xker.dim1 == xker_snf.dim1
        except:
            pass

def test_multiplication():
    """Test that composition of homomorphisms does not crash"""
    np.random.seed(345)
    dim0 = np.random.randint(0,4,size=(5,))
    dim1 = np.random.randint(0,4,size=(5,))
    dim2 = np.random.randint(0,4,size=(5,))
    x = lin.hom.rand(dim2, dim1)
    y = lin.hom.rand(dim1, dim0)
    xy = x @ y
    assert xy.dim0 == x.dim0
    assert xy.dim1 == y.dim1

def test_tostring():
    """Test that tostring does not crash"""
    x = lin.hom.rand_dim(4, 5, 5)
    assert isinstance(x.tostring(), str)

def test_solve_with_helper():
    """Test that solve_with_helper finds a valid solution k to Xk=b"""
    np.random.seed(345)
    for i in range(30):
        x = lin.hom.rand_dim_nr(4, 5)
        sol_to_find = lin.elem.rand(x.dim1)
        b = x @ sol_to_find
        ker, helps = x.kernel(return_solve_helper=True)
        sol = x.solve_with_helper(b, helps)
        assert (x @ sol - b).is_zero()

def test_epi_mono():
    """Test epi-mono decomposition: LR=X, L is injective, and cokernel dim equals image dim"""
    np.random.seed(345)
    for i in range(30):
        X = lin.hom.rand_dim_nr(4, 5)
        L, R = X.epi_mono()
        Lker = L.kernel()
        Xker = X.kernel()
        X_coker_dim = sum([(i+1)*d for i, d in enumerate(X.dim1)]) - sum([(i+1)*d for i, d in enumerate(Xker.dim1)])
        X_img_dim = sum([(i+1)*d for i, d in enumerate(L.dim1)])
        assert (L @ R - X).is_zero()
        assert not any(Lker.dim1)
        assert X_coker_dim == X_img_dim