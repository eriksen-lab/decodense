#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf, lo

import decodense

# init molecule
mol = gto.M(
    atom="""
        C  0.00000000     0.00000000    -0.06142879
        H  0.00000000     0.98919961     0.36571104
        H  0.00000000    -0.98919961     0.36571104
    """,
    verbose=0,
    output=None,
    basis="pcseg1",
    spin=2,
)

# mf calc
mf = scf.UKS(mol)
mf.xc = "pbe0"
mf.conv_tol = 1.0e-10
mf.kernel()

# occupied orbitals
alpha, beta = np.where(mf.mo_occ[0] > 0.0)[0], np.where(mf.mo_occ[1] > 0.0)[0]

# init mo coefficients
mo_coeff = []

# loop over spins
for i, spin_mo in enumerate((alpha, beta)):
    # pipek-mezey procedure
    loc = lo.PM(mol, mf.mo_coeff[i][:, spin_mo])
    loc.pop_method = "iao"
    loc.conv_tol = 1e-10
    mo_coeff.append(loc.kernel(mf.mo_coeff[i][:, spin_mo]))

    # jacobi sweep to ensure optimum is found
    mo_coeff[-1], isstable = loc.stability_jacobi(return_status=True)
    while not isstable:
        mo_coeff[-1] = loc.kernel(mo_coeff[-1])
        mo_coeff[-1], isstable = loc.stability_jacobi(return_status=True)

# decomposition
decomp = decodense.DecompCls(pop_method="iao", part="atoms", prop="dipole")
res = decodense.main(mol, decomp, mf, tuple(mo_coeff))

print(res)
