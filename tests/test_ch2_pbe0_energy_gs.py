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

# init molecule
mol = gto.M(verbose = 0, output = None, symmetry = True, basis = 'pcseg1', unit = 'au', spin = 2, atom = 'geom/ch2.xyz')

# mf calc
mf = scf.UKS(mol).density_fit(auxbasis='weigend', only_dfj=True)
mf.xc = 'pbe0'
mf = scf.fast_newton(mf, conv_tol=1.e-10)

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test(self):
        mf_e_tot = mf.e_tot
        for mo_basis in MO_BASIS:
            for mo_init in MO_INIT:
                for pop_method in POP_METHOD:
                    for part in PART:
                        with self.subTest(mo_basis=mo_basis, mo_init=mo_init, pop_method=pop_method, part=part):
                            decomp = decodense.DecompCls(mo_basis=mo_basis, mo_init=mo_init, pop_method=pop_method, part=part)
                            res = decodense.main(mol, decomp, mf)
                            e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                            self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_2(self):
        mf_e_tot = mf.e_tot
        mo_coeff = (mf.mo_coeff[0][:, mf.mo_occ[0] > 0.], mf.mo_coeff[1][:, mf.mo_occ[1] > 0.])
        mo_occ = (mf.mo_occ[0][mf.mo_occ[0] > 0.], mf.mo_occ[1][mf.mo_occ[1] > 0.])
        for part in PART:
            with self.subTest(part=part):
                decomp = decodense.DecompCls(part=part)
                res = decodense.main(mol, decomp, mf, mo_coeff=mo_coeff, mo_occ=mo_occ)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_3(self):
        mf_e_tot = mf.e_tot
        rdm1 = mf.make_rdm1()
        for part in PART:
            with self.subTest(part=part):
                decomp = decodense.DecompCls(part=part)
                res = decodense.main(mol, decomp, mf, rdm1_orb=rdm1)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_4(self):
        mf_e_tot = mf.e_tot
        rdm1 = mf.make_rdm1()
        for part in PART:
            with self.subTest(part=part):
                decomp = decodense.DecompCls(part=part)
                res = decodense.main(mol, decomp, mf, rdm1_eff=rdm1)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_5(self):
        mf_e_tot = mf.e_tot
        rdm1 = mf.make_rdm1()
        for part in PART:
            with self.subTest(part=part):
                decomp = decodense.DecompCls(part=part)
                res = decodense.main(mol, decomp, mf, rdm1_orb=rdm1, rdm1_eff=rdm1)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                self.assertAlmostEqual(mf_e_tot, e_tot, TOL)

if __name__ == '__main__':
    print('test: ch2_pbe0_energy_gs')
    unittest.main()

