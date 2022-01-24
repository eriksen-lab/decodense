#!/usr/bin/env python
# -*- coding: utf-8 -*

import unittest
import numpy as np
from pyscf import gto, scf, dft

import decodense

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
        for loc in LOC:
            for pop in POP:
                for part in PART:
                    with self.subTest(loc=loc, pop=pop, part=part):
                        decomp = decodense.DecompCls(loc=loc, pop=pop, part=part)
                        res = decodense.main(mol, decomp, mf)
                        self.assertAlmostEqual(mol.energy_nuc(), np.sum(res['struct']), 9)
                        self.assertAlmostEqual(mf.e_tot, np.sum(res['struct']) + np.sum(res['el']), 9)

if __name__ == '__main__':
    print('test: h2o_b3lyp')
    unittest.main()

