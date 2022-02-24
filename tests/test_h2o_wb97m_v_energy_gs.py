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

# init molecule
mol = gto.M(verbose = 0, output = None,
basis = 'pcseg1', symmetry = True,
atom = 'geom/h2o.xyz')

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
        for loc in LOC:
            for pop in POP:
                for part in PART:
                    with self.subTest(loc=loc, pop=pop, part=part):
                        decomp = decodense.DecompCls(loc=loc, pop=pop, part=part)
                        res = decodense.main(mol, decomp, mf)
                        if part == 'orbitals':
                            e_tot = np.sum(res['struct']) + np.sum(res['el'][0]) + np.sum(res['el'][1])
                        else:
                            e_tot = np.sum(res['struct']) + np.sum(res['el'])
                        self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_2(self):
        mf_e_tot = mf.e_tot
        rdm1 = mf.make_rdm1()
        for loc in LOC[:1]:
            for pop in POP[:1]:
                for part in PART[1:]:
                    with self.subTest(loc=loc, pop=pop, part=part):
                        decomp = decodense.DecompCls(loc=loc, pop=pop, part=part)
                        res = decodense.main(mol, decomp, mf, rdm1_orb=rdm1)
                        if part == 'orbitals':
                            e_tot= np.sum(res['struct']) + np.sum(res['el'][0]) + np.sum(res['el'][1])
                        else:
                            e_tot = np.sum(res['struct']) + np.sum(res['el'])
                        self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_3(self):
        mf_e_tot = mf.e_tot
        rdm1 = mf.make_rdm1()
        for loc in LOC[:1]:
            for pop in POP[:1]:
                for part in PART[1:]:
                    with self.subTest(loc=loc, pop=pop, part=part):
                        decomp = decodense.DecompCls(loc=loc, pop=pop, part=part)
                        res = decodense.main(mol, decomp, mf, rdm1_eff=rdm1)
                        if part == 'orbitals':
                            e_tot= np.sum(res['struct']) + np.sum(res['el'][0]) + np.sum(res['el'][1])
                        else:
                            e_tot = np.sum(res['struct']) + np.sum(res['el'])
                        self.assertAlmostEqual(mf_e_tot, e_tot, TOL)
    def test_4(self):
        mf_e_tot = mf.e_tot
        rdm1 = mf.make_rdm1()
        for loc in LOC[:1]:
            for pop in POP[:1]:
                for part in PART[1:]:
                    with self.subTest(loc=loc, pop=pop, part=part):
                        decomp = decodense.DecompCls(loc=loc, pop=pop, part=part)
                        res = decodense.main(mol, decomp, mf, rdm1_orb=rdm1, rdm1_eff=rdm1)
                        if part == 'orbitals':
                            e_tot= np.sum(res['struct']) + np.sum(res['el'][0]) + np.sum(res['el'][1])
                        else:
                            e_tot = np.sum(res['struct']) + np.sum(res['el'])
                        self.assertAlmostEqual(mf_e_tot, e_tot, TOL)

if __name__ == '__main__':
    print('test: h2o_wb97m_v_energy_gs')
    unittest.main()

