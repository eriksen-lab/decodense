#!/usr/bin/env python
# -*- coding: utf-8 -*

import unittest
import numpy as np
from pyscf import gto, scf, dft

import decodense

# decimal tolerance
TOL = 9

# settings
POP_METHOD = ("mulliken", "lowdin", "meta_lowdin", "becke", "iao")
PART = ("orbitals", "eda", "atoms")

# init molecule
mol = gto.M(verbose=0, output=None, basis="pcseg1", symmetry=True, atom="geom/h2o.xyz")

# mf calc
mf = dft.RKS(mol).density_fit(auxbasis="weigend", only_dfj=True)
mf.xc = "b3lyp"
mf = scf.fast_newton(mf, conv_tol=1.0e-10)

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.0)[0]

# mo coefficients
mo_coeff = 2 * (mf.mo_coeff[:, occ_mo],)


def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test(self):
        dipmom_tot = np.zeros(3)
        mf_dipmom_tot = mf.dip_moment(unit="au", verbose=0)
        for pop_method in POP_METHOD:
            for part in PART:
                with self.subTest(pop_method=pop_method, part=part):
                    decomp = decodense.DecompCls(
                        pop_method=pop_method, part=part, prop="dipole"
                    )
                    res = decodense.main(mol, decomp, mf, mo_coeff)
                    for ax_idx, axis in enumerate((" (x)", " (y)", " (z)")):
                        dipmom_tot[ax_idx] = np.sum(
                            res[decodense.decomp.CompKeys.tot + axis]
                        )
                    self.assertAlmostEqual(
                        np.linalg.norm(mf_dipmom_tot), np.linalg.norm(dipmom_tot), TOL
                    )


if __name__ == "__main__":
    print("test: h2o_b3lyp_dipmom_gs")
    unittest.main()
