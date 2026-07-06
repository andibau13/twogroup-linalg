# twogroup-linalg

Linear algebra utilities for finite abelian 2-groups. Intended to be fast for large 2-group consisting of many factors $\mathbb{Z}_{2^i}$ for small $i$.

It provides basic operations on homomorphisms between 2-groups:
- Class ```Hom``` representing homomorphism between 2-groups.
- Class ```Elem``` representing element of a 2-group.
- Zero, random, identity generators for ```Hom``` and ```Elem```.
- ```tostring``` method for ```Hom``` and ```Elem```.
- Composition between homomorphisms and application of homomorphisms to elements via ```@```.
- ```Hom``` method ```kernel``` computing a kernel isomorphism $K$ of a 2-group homomorphism $A$. $K$ is an injective 2-group homomorphism such that $AK=0$.
- ```Hom``` method ```solve_with_helper```, solving a linear equation $Ak=b$ for a 2-group homomorphism $A$ and 2-group element $b$.
- ```Hom``` method ```epi_mono```, computing an epi-mono decomposition $(L,R)$ of a 2-group homomorphism $A$. $L$ is a surjective 2-group homomorphism and $R$ an injective 2-group homomorphism such that $A=LR$.


## Installation

```bash
pip install git+https://github.com/andibau13/twogroup-linalg.git
```

## Usage

```python
import twogroup_linalg as lin
group0 = [2,2,3,4] # group Z2^2 x Z4^3 x Z8^4
group1 = [2,5,3]
group2 = [0,8,3]
x = lin.Hom.rand(group1, group0) # homomorphism group0 -> group1
y = lin.Hom.rand(group2, group1)
a = lin.Elem.rand(group0)
print(f"x:\n{x.tostring()}\ny:\n{y.tostring()}\na:\n{a.tostring()}\n")
print(f"Composition yx:\n{(y @ x).tostring()}\n")
print(f"Application x(a):\n{(x @ a).tostring()}\n")
ker_x, helper = x.kernel(return_solve_helper=True)
print(f"Kernel of x:\n{ker_x.tostring()}\n") # columns represent kernel elements
print(f"Solution to x(k)=x(a)\n{(x.solve_with_helper(x @ a, helper)).tostring()}\n")
l, r = x.epi_mono()
print(f"Epi-mono: x=lr with l=\n{l.tostring()}\nand r=\n{r.tostring()}")
```