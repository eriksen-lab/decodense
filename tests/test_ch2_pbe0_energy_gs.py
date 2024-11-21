#!/usr/bin/env python
# -*- coding: utf-8 -*

import unittest
import numpy as np
from pyscf import gto, scf

import decodense

# decimal tolerance
TOL = 9

# settings
POP_METHOD = ("mulliken", "lowdin", "meta_lowdin", "becke", "iao")
PART = ("orbitals", "eda", "atoms")

# init molecule
mol = gto.M(
    verbose=0,
    output=None,
    symmetry=True,
    basis="pcseg1",
    unit="au",
    spin=2,
    atom="geom/ch2.xyz",
)

# mf calc
mf = scf.UKS(mol).density_fit(auxbasis="weigend", only_dfj=True)
mf.xc = "pbe0"
mf = scf.fast_newton(mf, conv_tol=1.0e-10)

# occupied orbitals
alpha = np.where(mf.mo_occ[0] > 0.0)[0]
beta = np.where(mf.mo_occ[1] > 0.0)[0]

# mo coefficients
mo_coeff = (mf.mo_coeff[0][:, alpha], mf.mo_coeff[1][:, beta])


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


if __name__ == "__main__":
    print("test: ch2_pbe0_energy_gs")
    unittest.main()
