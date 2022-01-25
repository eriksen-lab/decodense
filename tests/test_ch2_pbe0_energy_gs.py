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

if __name__ == '__main__':
    print('test: ch2_pbe0_energy_gs')
    unittest.main()

