#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf, lo
from pyscf.opentrustregion import to_otr

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
mf = to_otr(scf.RKS(mol, xc="pbe0"))
mf.conv_tol = 1.0e-10
mf.kernel()

# verify SCF solution is a true minimum
stable, direction = mf.stability_check()

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.0)[0]

# pipek-mezey procedure with OTR
loc = to_otr(lo.PipekMezey(mol, mf.mo_coeff[:, occ_mo]))
loc.pop_method = "iao"
loc.conv_tol = 1e-10
loc.line_search = True
mo_coeff = loc.kernel()

# verify optimum is a true minimum
stable, direction = loc.stability_check()

# decomposition
decomp = decodense.DecompCls(part="orbitals")
res = decodense.main(mol, decomp, mf, mo_coeff)

print(res)
