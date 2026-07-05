import twogroup_linalg as lin
import flint_wrappers as fl
import pari_wrappers as pa
import numpy as np

int_kernel = fl.int_kernel
snf = pa.snf
int_inv = fl.int_inv

# calculate the kernel of a homomorphism Z_2^dim1[0] x Z_4^dim1[1] x Z_8^dim1[2] -> Z_2^dim0[0] x Z_4^dim0[1] x Z_8^dim0[2] using kernels and smith normal form of integer matrices
def z248_z248_kernel(self):
    modlist0 = np.array([2]*self.dim0[0]+[4]*self.dim0[1]+[8]*self.dim0[2])
    modlist1 = np.array([2]*self.dim1[0]+[4]*self.dim1[1]+[8]*self.dim1[2])
    A_extended = np.hstack([self.enhanced(), np.diag(modlist0)])
    A_extended_kernel = int_kernel(A_extended)
    A_mod_kernel = A_extended_kernel[:self.M.shape[1],:]
    A_mod_kernel_extended = np.hstack([A_mod_kernel, np.diag(modlist1)])
    A_modlattice = int_kernel(A_mod_kernel_extended)[:A_mod_kernel.shape[1],:]
    U, _, D = snf(A_modlattice)
    #Vinv = int_inv(V)
    Uinv = int_inv(U)
    if D.shape[0]!=D.shape[1]:
        print(D)
        raise ValueError("D is not square. this probably doesn't work if D is not square!!")
    D_diag = D.diagonal()
    def nr_geq_entries(i):
        leq_entries = (D_diag < i)
        return int(leq_entries.argmax()) if leq_entries.any() else len(D_diag)
    nr_geq8_entries = nr_geq_entries(8)
    nr_geq4_entries = nr_geq_entries(4)
    nr_geq2_entries = nr_geq_entries(2)
    full_kernel = A_mod_kernel@Uinv
    mod8_kernel = full_kernel[:,:nr_geq8_entries]
    mod4_kernel = full_kernel[:, nr_geq8_entries:nr_geq4_entries]
    mod2_kernel = full_kernel[:, nr_geq4_entries:nr_geq2_entries]
    dim2 = (nr_geq2_entries - nr_geq4_entries, nr_geq4_entries - nr_geq8_entries, nr_geq8_entries)
    final_kernel = lin.z248_hom(np.hstack([mod2_kernel, mod4_kernel, mod8_kernel]), self.dim1, dim2)
    final_kernel.M = final_kernel.unenhanced()
    final_kernel.reduce_mod()
    #kernel_reorganize = reduce_mod(z2z4z8_unenhance(np.hstack([mod2_kernel, mod4_kernel, mod8_kernel]), dim1, dim2), dim1, dim2)
    return final_kernel

def twogroup_kernel_snf(self):
    def modlist(dim):
        return np.array([2**(i+1) for i, d in enumerate(dim) for _ in range(d)], dtype=int)

    modlist0 = modlist(self.dim0)
    modlist1 = modlist(self.dim1)
    domain_size = int(sum(self.dim1))
    codomain_size = int(sum(self.dim0))
    kernel_group_len = len(self.dim1)

    if domain_size == 0:
        final_kernel = type(self)(np.zeros((0, 0), dtype=int), self.dim1, [0]*kernel_group_len)
        final_kernel.reduce_mod()
        return final_kernel

    if codomain_size == 0:
        final_kernel = type(self)(np.eye(domain_size, dtype=int), self.dim1, self.dim1)
        final_kernel.reduce_mod()
        return final_kernel

    A_extended = np.hstack([self.enhanced().M, np.diag(modlist0)])
    A_extended_kernel = int_kernel(A_extended)
    A_mod_kernel = A_extended_kernel[:self.M.shape[1],:]
    A_mod_kernel_extended = np.hstack([A_mod_kernel, np.diag(modlist1)])
    A_modlattice = int_kernel(A_mod_kernel_extended)[:A_mod_kernel.shape[1],:]
    U, _, D = snf(A_modlattice)
    Uinv = int_inv(U)
    if D.shape[0] != D.shape[1]:
        print(D)
        raise ValueError("D is not square. this probably doesn't work if D is not square!!")
    D_diag = np.abs(D.diagonal())
    invariant_moduli = {2**(i+1) for i in range(kernel_group_len)}
    bad_invariants = [d for d in D_diag if d != 1 and d not in invariant_moduli]
    if bad_invariants:
        raise ValueError(f"Kernel has unexpected invariant factors: {bad_invariants}")

    full_kernel = A_mod_kernel @ Uinv
    kernel_blocks = [full_kernel[:, D_diag == 2**(i+1)] for i in range(kernel_group_len)]
    dim2 = [block.shape[1] for block in kernel_blocks]
    if kernel_blocks:
        kernel_matrix = np.hstack(kernel_blocks)
    else:
        kernel_matrix = np.zeros((domain_size, 0), dtype=int)

    final_kernel = type(self)(kernel_matrix, self.dim1, dim2)
    final_kernel = final_kernel.unenhanced()
    final_kernel.reduce_mod()
    return final_kernel
