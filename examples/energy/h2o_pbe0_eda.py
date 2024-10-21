#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf

import decodense

# init molecule
mol = gto.M(
    atom="""
        O  0.00000000  0.00000000  0.00000000
        H -0.75390364  0.00000000 -0.58783729
        H  0.75390364  0.00000000 -0.58783729
    """,
    verbose=0,
    output=None,
    basis="pcseg1",
)

# mf calc
mf = scf.RKS(mol)
mf.xc = "pbe0"
mf.conv_tol = 1.0e-10
mf.kernel()

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.0)[0]

# mo coefficients
mo_coeff = 2 * (mf.mo_coeff[:, occ_mo],)

# decomposition
decomp = decodense.DecompCls(part="eda")
res = decodense.main(mol, decomp, mf, mo_coeff)

print(res)
