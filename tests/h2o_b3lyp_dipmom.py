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
basis = 'ccpvdz', symmetry = True,
atom = 'h2o.xyz')

mf = dft.RKS(mol).density_fit(auxbasis='weigend', only_dfj=True)
mf.xc = 'b3lyp'
mf = scf.fast_newton(mf, conv_tol=1.e-10)

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test(self):
        mf_dipmom_tot = mf.dip_moment(unit='au', verbose=0)
        for loc in LOC:
            for pop in POP:
                for part in PART:
                    with self.subTest(loc=loc, pop=pop, part=part):
                        decomp = decodense.DecompCls(loc=loc, pop=pop, part=part, prop='dipole')
                        res = decodense.main(mol, decomp, mf)
                        if part == 'orbitals':
                            dipmom_tot = np.fromiter(map(np.sum, res['el'][0].T), dtype=np.float64, count=3) \
                                          + np.fromiter(map(np.sum, res['el'][1].T), dtype=np.float64, count=3) \
                                          + np.sum(res['struct'], axis=0)
                        else:
                            dipmom_tot = np.fromiter(map(np.sum, res['el'].T), dtype=np.float64, count=3) \
                                          + np.fromiter(map(np.sum, res['struct'].T), dtype=np.float64, count=3)
                        np.testing.assert_array_almost_equal(mf_dipmom_tot, dipmom_tot, TOL)

if __name__ == '__main__':
    print('test: h2o_b3lyp_dipmom')
    unittest.main()

