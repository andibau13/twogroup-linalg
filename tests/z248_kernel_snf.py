import third_order_gates.z248_linalg as lin
import third_order_gates.flint_wrappers as fl
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

