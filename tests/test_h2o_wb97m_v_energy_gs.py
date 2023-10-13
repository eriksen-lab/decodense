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
mol = gto.M(verbose = 0, output = None, basis = 'pcseg1', symmetry = True, atom = 'geom/h2o.xyz')

# mf calc
mf = dft.RKS(mol)
mf.xc = 'wb97m_v'
mf.nlc = 'vv10'
mf.nlcgrids.atom_grid = (50, 194)
mf.nlcgrids.prune = dft.gen_grid.sg1_prune
mf.kernel()

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
        mo_coeff = (mf.mo_coeff[:, mf.mo_occ > 0.],) * 2
        mo_occ = (mf.mo_occ[mf.mo_occ > 0.] / 2.,) * 2
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
    print('test: test_h2o_wb97m_v_energy_gs')
    unittest.main()

