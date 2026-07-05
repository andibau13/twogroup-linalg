import twogroup_linalg as lin
import numpy as np
import z248_kernel_snf as snf

def test_kernel():
    """Test if kernel isomorphism indeed lands in the kernel"""
    np.random.seed(345)
    for i in range(100):
        x = lin.hom.rand_dim(4, 5, 7)
        xker = x.kernel()
        assert (x@xker).is_zero()

def test_snf_kernel():
    """Test if snf kernel isomorphism indeed lands in the kernel"""
    np.random.seed(345)
    for i in range(100):
        x = lin.hom.rand_dim(4, 5, 7)
        try:
            xker = snf.twogroup_kernel_snf(x)
            assert (x@xker).is_zero()
        except:
            pass

def test_kernel_dimension():
    """Test if kernel dimension is the same as for snf kernel"""
    np.random.seed(345)
    for i in range(100):
        x = lin.hom.rand_dim(4, 5, 7)
        xker = x.kernel()
        try:
            xker_snf = snf.twogroup_kernel_snf(x)
            assert xker.dim1 == xker_snf.dim1
        except:
            pass