#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf, qmmm, lo
from pyscf.opentrustregion import to_otr

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
# TODO: above line raises AttributeError: module 'pyscf.qmmm' has no attribute 'add_mmpol'
# see also issue https://github.com/eriksen-lab/decodense/issues/31

# run mf
mf.kernel()

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.0)[0]

# pipek-mezey procedure with OTR
loc = to_otr(lo.PipekMezey(mol, mf.mo_coeff[:, occ_mo]))
# TODO: verify whether to_otr works correctly once the earlier TODO is fixed
loc.pop_method = "iao"
loc.conv_tol = 1e-10
loc.line_search = True
mo_coeff = loc.kernel()

# verify optimum is a true minimum
stable, direction = loc.stability_check()

# decomposition
decomp = decodense.DecompCls(pop_method="iao", part="orbitals")
res = decodense.main(mol, decomp, mf, mo_coeff)

print(res)
