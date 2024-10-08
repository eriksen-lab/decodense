#!/usr/bin/env python
# -*- coding: utf-8 -*

import unittest
import numpy as np
from pyscf import gto, scf, dft

import decodense

# decimal tolerance
TOL = 9

# settings
MO_BASIS = ('can', 'fb', 'pm')
MO_INIT = ('can', 'cholesky', 'ibo')
POP_METHOD = ('mulliken', 'lowdin', 'meta_lowdin', 'becke', 'iao')
LOC_EXP = (2, 4)
PART = ('orbitals', 'eda', 'atoms')

# 1a2 state
OCC_IDX, VIRT_IDX = 7, 8

def format_mf(mf):
    mo_coeff = np.asarray((mf.mo_coeff,) * 2)
    mo_occ = np.asarray((np.zeros(mf.mo_occ.size, dtype=np.float64),) * 2)
    mo_occ[0][np.where(0. < mf.mo_occ)] += 1.
    mo_occ[1][np.where(1. < mf.mo_occ)] += 1.
    return mo_coeff, mo_occ

def gs_calc(mol):
    mf_gs = dft.RKS(mol)
    mf_gs.xc = 'camb3lyp'
    mf_gs.conv_tol = 1.e-10
    mf_gs.kernel()
    return mf_gs

def ex_calc(mol, mo_coeff, mo_occ):
    mo_occ[0][OCC_IDX] = 0.
    mo_occ[0][VIRT_IDX] = 1.
    mf_ex = dft.UKS(mol)
    mf_ex.xc = 'camb3lyp'
    mf_ex.conv_tol = 1.e-10
    mf_ex = scf.addons.mom_occ(mf_ex, mo_coeff, mo_occ)
    dm = mf_ex.make_rdm1(mo_coeff, mo_occ)
    mf_ex.kernel(dm)
    return mf_ex

# init mol
mol = gto.M(verbose = 0, output = None, symmetry = True, basis = 'pcseg1', atom = 'geom/ch2o.xyz')

# ground-state mf calc
mf_gs = gs_calc(mol)
c_gs, mo_occ_gs = format_mf(mf_gs)

# excited-state mf calc
mf_ex = ex_calc(mol, c_gs, mo_occ_gs)

def tearDownModule():
    global mol, mf_gs, mf_ex
    mol.stdout.close()
    del mol, mf_gs, mf_ex

class KnownValues(unittest.TestCase):
    def test(self):
        mf_e_tot = mf_ex.e_tot
        for mo_basis in MO_BASIS:
            for mo_init in MO_INIT:
                for pop_method in POP_METHOD:
                    for part in PART:
                        with self.subTest(mo_basis=mo_basis, mo_init=mo_init, pop_method=pop_method, part=part):
                            decomp = decodense.DecompCls(mo_basis=mo_basis, mo_init=mo_init, pop_method=pop_method, part=part)
                            res = decodense.main(mol, decomp, mf_ex)
                            e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                            self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_2(self):
        mf_e_tot = mf_ex.e_tot
        mo_coeff = (mf_ex.mo_coeff[0][:, mf_ex.mo_occ[0] > 0.], mf_ex.mo_coeff[1][:, mf_ex.mo_occ[1] > 0.])
        mo_occ = (mf_ex.mo_occ[0][mf_ex.mo_occ[0] > 0.], mf_ex.mo_occ[1][mf_ex.mo_occ[1] > 0.])
        for part in PART:
            with self.subTest(part=part):
                decomp = decodense.DecompCls(part=part)
                res = decodense.main(mol, decomp, mf_ex, mo_coeff=mo_coeff, mo_occ=mo_occ)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_3(self):
        mf_e_tot = mf_ex.e_tot
        rdm1 = mf_ex.make_rdm1()
        for part in PART:
            with self.subTest(part=part):
                decomp = decodense.DecompCls(part=part)
                res = decodense.main(mol, decomp, mf_ex, rdm1_orb=rdm1)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_4(self):
        mf_e_tot = mf_ex.e_tot
        rdm1 = mf_ex.make_rdm1()
        for part in PART:
            with self.subTest(part=part):
                decomp = decodense.DecompCls(part=part)
                res = decodense.main(mol, decomp, mf_ex, rdm1_eff=rdm1)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_5(self):
        mf_e_tot = mf_ex.e_tot
        rdm1 = mf_ex.make_rdm1()
        for part in PART:
            with self.subTest(part=part):
                decomp = decodense.DecompCls(part=part)
                res = decodense.main(mol, decomp, mf_ex, rdm1_orb=rdm1, rdm1_eff=rdm1)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                self.assertAlmostEqual(mf_e_tot, e_tot, TOL)

if __name__ == '__main__':
    print('test: test_ch2o_camb3lyp_energy_ex.py')
    unittest.main()

