#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf, lo

import decodense

# init molecule
mol = gto.M(
    atom = """
        C  0.00000000     0.00000000    -0.06142879
        H  0.00000000     0.98919961     0.36571104
        H  0.00000000    -0.98919961     0.36571104
    """,
    verbose = 0, 
    output = None, 
    basis = "pcseg1", 
    spin = 2, 
)

# ground state mf calc
mf_gs = scf.ROHF(mol)
mf_gs.conv_tol = 1.e-10
mf_gs.kernel()
rdm1_gs = mf_gs.make_rdm1()

# convert mo coefficients and mo occupation to unrestricted quantities
mo_coeff = np.asarray((mf_gs.mo_coeff,) * 2)
mo_occ = np.asarray((np.zeros(mf_gs.mo_occ.size, dtype=np.float64),) * 2)
mo_occ[0][np.where(0. < mf_gs.mo_occ)] += 1.
mo_occ[1][np.where(1. < mf_gs.mo_occ)] += 1.

# excited state mf calc
mo_occ[0][4] = 0.
mo_occ[0][5] = 1.
mf_ex = scf.ROHF(mol).density_fit(auxbasis='weigend', only_dfj=True)
mf_ex.conv_tol = 1.e-10
mf_ex = scf.addons.mom_occ(mf_ex, mo_coeff[0], mo_occ)
dm = mf_ex.make_rdm1(mo_coeff[0], mo_occ[0] + mo_occ[1])
mf_ex.kernel(dm)
rdm1_ex = mf_ex.make_rdm1()

# dm_sum & dm_delta
rdm1_sum = rdm1_ex + rdm1_gs
rdm1_delta = rdm1_ex - rdm1_gs

# get mo coefficients as unrestricted quantities
c = np.asarray((mf_ex.mo_coeff,) * 2)

# overlap matrix
s = mol.intor_symmetric('int1e_ovlp')

# ao to mo transformation of dm difference
rdm1_mo = np.einsum('xpi,pq,xqr,rs,xsj->xij', c, s, rdm1_delta, s, c)

# diagonalize dm difference to get natural difference orbitals
occ_no, u = np.linalg.eigh(rdm1_mo)

# transform to ndo basis
mo_no = np.einsum('xip,xpj->xij', c, u)

# retain only significant ndos
mo_coeff = (mo_no[0][:, np.where(np.abs(occ_no[0]) >= 1.e-12)[0]], mo_no[1][:, np.where(np.abs(occ_no[1]) >= 1.e-12)[0]])     
mo_occ = (occ_no[0][np.where(np.abs(occ_no[0]) >= 1.e-12)], occ_no[1][np.where(np.abs(occ_no[1]) >= 1.e-12)])

# decomposition
decomp = decodense.DecompCls(pop_method="mulliken", part="atoms", ndo=True)
res = decodense.main(mol, decomp, mf_ex, mo_coeff, mo_occ, rdm1=rdm1_sum)

print(res)
