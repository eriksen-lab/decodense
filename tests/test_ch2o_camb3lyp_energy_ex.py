#!/usr/bin/env python
# -*- coding: utf-8 -*

import unittest
import numpy as np
from pyscf import gto, scf, dft

import decodense

# decimal tolerance
TOL = 9

# settings
POP_METHOD = ('mulliken', 'lowdin', 'meta_lowdin', 'becke', 'iao')
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

# occupied orbitals
alpha = np.where(mf_ex.mo_occ[0] > 0.)[0]
beta = np.where(mf_ex.mo_occ[1] > 0.)[0]

# mo coefficients
mo_coeff = (mf_ex.mo_coeff[0][:, alpha], mf_ex.mo_coeff[1][:, beta])

def tearDownModule():
    global mol, mf_gs, mf_ex
    mol.stdout.close()
    del mol, mf_gs, mf_ex

class KnownValues(unittest.TestCase):
    def test(self):
        mf_e_tot = mf_ex.e_tot
        for pop_method in POP_METHOD:
            for part in PART:
                with self.subTest(pop_method=pop_method, part=part):
                    decomp = decodense.DecompCls(pop_method=pop_method, part=part)
                    res = decodense.main(mol, decomp, mf_ex, mo_coeff)
                    e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                    self.assertAlmostEqual(mf_e_tot, e_tot, TOL)

if __name__ == '__main__':
    print('test: test_ch2o_camb3lyp_energy_ex.py')
    unittest.main()

