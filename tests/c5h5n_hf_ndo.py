#!/usr/bin/env python
# -*- coding: utf-8 -*

import unittest
import numpy as np
from pyscf import gto, scf, dft

import decodense

# decimal tolerance
TOL = 9

# settings
LOC = ('', 'fb', 'pm', 'ibo-2', 'ibo-4')
POP = ('mulliken', 'iao')
PART = ('orbitals', 'eda', 'atoms')

# 1a2 state
OCC_IDX, VIRT_IDX = 18, 22

def gs_calc(mol):
    mf_gs = scf.RHF(mol).density_fit(auxbasis='weigend', only_dfj=True)
    return scf.fast_newton(mf_gs, conv_tol=1.e-10)

def ex_calc(mol, mo_coeff, mo_occ):
    mo_occ[0][OCC_IDX] = 0.
    mo_occ[0][VIRT_IDX] = 1.
    mf_ex = scf.UHF(mol).density_fit(auxbasis='weigend', only_dfj=True)
    mf_ex.conv_tol = 1.e-10
    mf_ex = scf.addons.mom_occ(mf_ex, mo_coeff, mo_occ)
    dm = mf_ex.make_rdm1(mo_coeff, mo_occ)
    mf_ex.kernel(dm)
    return mf_ex

# init mol
mol = gto.M(verbose = 0, output = None, symmetry = True, basis = 'pcseg1', atom = 'geom/c5h5n.xyz')

# ground-state mf calc
mf_gs = gs_calc(mol)
mf_gs = mf_gs.to_uhf()
rdm1_gs = mf_gs.make_rdm1()

# excited-state mf calc
mf_ex = ex_calc(mol, mf_gs.mo_coeff, mf_gs.mo_occ)
rdm1_ex = mf_ex.make_rdm1()

# dm_sum & dm_delta
rdm1_sum = rdm1_ex + rdm1_gs
rdm1_delta = rdm1_ex - rdm1_gs

def tearDownModule():
    global mol, mf_gs, mf_ex
    mol.stdout.close()
    del mol, mf_gs, mf_ex

class KnownValues(unittest.TestCase):
    def test(self):
        mf_e_tot = mf_ex.e_tot - mf_gs.e_tot
        for loc in LOC:
            for pop in POP:
                for part in PART:
                    with self.subTest(loc=loc, pop=pop, part=part):
                        decomp = decodense.DecompCls(loc=loc, pop=pop, part=part, ndo=True)
                        if loc == '':
                            if pop == 'mulliken':
                                if part == 'orbitals':
                                    res = decodense.main(mol, decomp, mf_ex, rdm1_orb=rdm1_delta, rdm1_eff=rdm1_sum)
                                    e_tot = np.sum(res['el'][0]) + np.sum(res['el'][1])
                                    self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
                        else:
                            with self.assertRaises(NotImplementedError):
                                res = decodense.main(mol, decomp, mf_ex, rdm1_orb=rdm1_delta, rdm1_eff=rdm1_sum)

if __name__ == '__main__':
    print('test: c5h5n_hf_ndo')
    unittest.main()

