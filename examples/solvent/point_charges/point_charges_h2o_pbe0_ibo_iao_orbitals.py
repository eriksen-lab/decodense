#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf, qmmm
from pyscf.opentrustregion import PipekMezeyOTR, mf_to_otr

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

# mf settings
mf = scf.RKS(mol, xc="pbe0")
mf.conv_tol = 1.0e-10

# randomly add point charges, then apply mf_to_otr
np.random.seed(1)
coords = np.random.random((5, 3)) * 10
charges = (np.arange(5) + 1.0) * -0.1
mf = mf_to_otr(qmmm.mm_charge(mf, coords, charges))

# run mf
mf.kernel()

# verify SCF solution is a true minimum
stable, direction = mf.stability_check()

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.0)[0]

# pipek-mezey procedure with OTR
loc = PipekMezeyOTR(mol, mf.mo_coeff[:, occ_mo])
loc.pop_method = "iao"
loc.conv_tol = 1e-10
mo_coeff = loc.kernel()

# verify optimum is a true minimum
stable, direction = loc.stability_check()

# decomposition
decomp = decodense.DecompCls(pop_method="iao", part="orbitals")
res = decodense.main(mol, decomp, mf, mo_coeff)

print(res)
