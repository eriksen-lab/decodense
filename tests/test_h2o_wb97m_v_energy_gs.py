#!/usr/bin/env python
# -*- coding: utf-8 -*

import unittest
import numpy as np
from pyscf import gto, dft

import decodense

# decimal tolerance
TOL = 9

# settings
POP_METHOD = ('mulliken', 'lowdin', 'meta_lowdin', 'becke', 'iao')
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

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.)[0]

# mo coefficients
mo_coeff = 2 * (mf.mo_coeff[:, occ_mo], )

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test(self):
        mf_e_tot = mf.e_tot
        for pop_method in POP_METHOD:
            for part in PART:
                with self.subTest(pop_method=pop_method, part=part):
                    decomp = decodense.DecompCls(pop_method=pop_method, part=part)
                    res = decodense.main(mol, decomp, mf, mo_coeff)
                    e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                    self.assertAlmostEqual(mf_e_tot, e_tot, TOL)

if __name__ == '__main__':
    print('test: test_h2o_wb97m_v_energy_gs')
    unittest.main()

