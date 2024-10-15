#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf

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
alpha = np.where(mf.mo_occ[0] > 0.0)[0]
beta = np.where(mf.mo_occ[1] > 0.0)[0]

# mo coefficients
mo_coeff = (mf.mo_coeff[0][:, alpha], mf.mo_coeff[1][:, beta])

# decomposition
decomp = decodense.DecompCls(part="eda")
res = decodense.main(mol, decomp, mf, mo_coeff)

print(res)
