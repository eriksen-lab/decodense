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
OCC_IDX, VIRT_IDX = 4, 5

def format_mf(mf):
    mo_coeff = np.asarray((mf.mo_coeff,) * 2)
    mo_occ = np.asarray((np.zeros(mf.mo_occ.size, dtype=np.float64),) * 2)
    mo_occ[0][np.where(0. < mf.mo_occ)] += 1.
    mo_occ[1][np.where(1. < mf.mo_occ)] += 1.
    dm = mf.make_rdm1()
    return mo_coeff, mo_occ, dm

def gs_calc(mol):
    mf_gs = scf.ROHF(mol).density_fit(auxbasis='weigend', only_dfj=True)
    return scf.fast_newton(mf_gs, conv_tol=1.e-10)

def ex_calc(mol, mo_coeff, mo_occ):
    mo_occ[0][OCC_IDX] = 0.
    mo_occ[0][VIRT_IDX] = 1.
    mf_ex = scf.ROHF(mol).density_fit(auxbasis='weigend', only_dfj=True)
    mf_ex.conv_tol = 1.e-10
    mf_ex = scf.addons.mom_occ(mf_ex, mo_coeff[0], mo_occ)
    dm = mf_ex.make_rdm1(mo_coeff[0], mo_occ[0] + mo_occ[1])
    mf_ex.kernel(dm)
    return mf_ex

# init mol
mol = gto.M(verbose = 0, output = None, symmetry = True, basis = 'pcseg1', unit = 'au', spin = 2, atom = 'geom/ch2.xyz')

# ground-state mf calc
mf_gs = gs_calc(mol)
c_gs, mo_occ_gs, rdm1_gs = format_mf(mf_gs)

# excited-state mf calc
mf_ex = ex_calc(mol, c_gs, mo_occ_gs)
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
    print('test: ch2_hf_ndo')
    unittest.main()

