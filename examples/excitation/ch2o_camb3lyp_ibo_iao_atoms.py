#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf, lo

import decodense

# init molecule
mol = gto.M(
    atom = """
        O     0.598112222862  -0.000014392138  -0.000007440073
        C    -0.593888140019   0.000049662540  -0.000000040694
        H    -1.202104656541   0.240676789231  -0.892725593892
        H    -1.202119426406  -0.240512059699   0.892733074659
    """,
    verbose = 0, 
    output = None, 
    basis = "pcseg1", 
    symmetry=True,
)

# ground state mf calculation
mf_gs = scf.RKS(mol)
mf_gs.xc = "camb3lyp"
mf_gs.conv_tol = 1.e-10
mf_gs.kernel()

# convert mo coefficients and mo occupation to unrestricted quantities
mo_coeff = np.asarray((mf_gs.mo_coeff,) * 2)
mo_occ = np.asarray((np.zeros(mf_gs.mo_occ.size, dtype=np.float64),) * 2)
mo_occ[0][np.where(0. < mf_gs.mo_occ)] += 1.
mo_occ[1][np.where(1. < mf_gs.mo_occ)] += 1.

# target 1a2 state
mo_occ[0][7] = 0.
mo_occ[0][8] = 1.

# excited state mf calculation
mf_ex = scf.UKS(mol)
mf_ex.xc = 'camb3lyp'
mf_ex.conv_tol = 1.e-10
mf_ex = scf.addons.mom_occ(mf_ex, mo_coeff, mo_occ)
dm = mf_ex.make_rdm1(mo_coeff, mo_occ)
mf_ex.kernel(dm)

# occupied orbitals
alpha, beta = np.where(mf_ex.mo_occ[0] > 0.)[0], np.where(mf_ex.mo_occ[1] > 0.)[0]

# init mo coefficients
mo_coeff = (np.zeros_like(mf_ex.mo_coeff[0]), np.zeros_like(mf_ex.mo_coeff[1]))

# loop over spins
for i, spin_mo in enumerate((alpha, beta)):
    # pipek-mezey procedure
    # create mock object to circumvent pyscf issue #1896
    mock_mf = mf_ex.copy()
    mock_mf.mo_coeff = mf_ex.mo_coeff[i]
    mock_mf.mo_occ = mf_ex.mo_occ[i]
    loc = lo.PM(mol, mf=mock_mf)
    loc.pop_method = "iao"
    loc.conv_tol = 1e-10
    mo_coeff[i][:, spin_mo] = loc.kernel(mf_ex.mo_coeff[i][:, spin_mo])

    # jacobi sweep to ensure optimum is found
    isstable, mo_coeff[i][:, spin_mo] = loc.stability_jacobi()
    while not isstable:
        mo_coeff[i][:, spin_mo] = loc.kernel(mo_coeff[i][:, spin_mo])
        isstable, mo_coeff[i][:, spin_mo] = loc.stability_jacobi()

# decomposition
decomp = decodense.DecompCls(pop_method="iao", part="atoms")
res = decodense.main(mol, decomp, mf_ex, mo_coeff)

print(res)
    