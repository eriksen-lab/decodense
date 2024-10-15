#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import scf, lo
from pyscf.pbc import df, dft, gto
from pyscf.pbc.tools.k2gamma import k2gamma, to_supercell_ao_integrals

import decodense

# init cell
cell = gto.Cell(
    atom="""
        O  0.00000000  0.00000000  0.00000000
        H -0.75390364  0.00000000 -0.58783729
        H  0.75390364  0.00000000 -0.58783729
    """,
    verbose=0,
    output=None,
    basis="gth-szv-molopt-sr",
    pseudo="gth-pbe",
    a=4 * np.eye(3),
    exp_to_discard=0.1,
)
cell.build()

# number of k-points / cell images in each direction
nkpt = 2
kmesh = [nkpt, nkpt, nkpt]
kpts = cell.make_kpts(kmesh)

# run k-point calculation
kmf = dft.KRKS(cell, kpts).density_fit(auxbasis="weigend")
kmf.xc = "pbe,pbe"
edft = kmf.kernel()

# transform the k-point mf object to mf object for a supercell at gamma-point
mf_scf = k2gamma(kmf)
supcell, mo_coeff, mo_occ, e_mo = (
    mf_scf.mol,
    mf_scf.mo_coeff,
    mf_scf.mo_occ,
    mf_scf.mo_energy,
)
j_int = to_supercell_ao_integrals(cell, kpts, kmf.get_j())
mf = dft.RKS(supcell).density_fit().apply(scf.addons.remove_linear_dep_)
mf.xc = "pbe,pbe"
mf.with_df = df.df.DF(supcell)
mf.with_df.auxbasis = "weigend"
mf.mo_coeff, mf.mo_occ, mf.mo_energy = mo_coeff, mo_occ, e_mo
mf.initialize_grids(supcell, mf.make_rdm1(), supcell.make_kpts([1, 1, 1]))
mf.vj = j_int

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.0)[0]

# pipek-mezey procedure
loc = lo.PM(supcell, mf=mf)
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
res = decodense.main(supcell, decomp, mf, mo_coeff)

print(res)
