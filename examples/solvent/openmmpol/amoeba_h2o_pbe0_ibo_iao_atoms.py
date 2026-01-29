#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf, lo, qmmm

import decodense

# read in molecule information from openmmpol qm file
file = "four_water"
with open(f"{file}_qm.xyz") as f:
    molstr = f.read()

# init molecule
mol = gto.M(
    atom=molstr[60:],
    verbose=0,
    output=None,
    basis="pcseg1",
)

# mf calc
mf = scf.RKS(mol)
mf.xc = "pbe0"
mf.conv_tol = 1.0e-10

# add openmmpol mm region
mf = qmmm.add_mmpol(mf, f"{file}_si.json", use_si_qm_coord=True)

# run mf
mf.kernel()

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.0)[0]

# pipek-mezey procedure
loc = lo.PM(mol, mf=mf)
loc.pop_method = "iao"
loc.conv_tol = 1e-10
mo_coeff = loc.kernel(mf.mo_coeff[:, occ_mo])

# jacobi sweep to ensure optimum is found
isstable, mo_coeff = loc.stability_jacobi()
while not isstable:
    mo_coeff = loc.kernel(mo_coeff)
    isstable, mo_coeff = loc.stability_jacobi()

# decomposition
decomp = decodense.DecompCls(pop_method="iao", part="atoms")
res = decodense.main(mol, decomp, mf, mo_coeff)

print(res)
