#!/usr/bin/env python
# -*- coding: utf-8 -*

import unittest
import numpy as np
from pyscf import scf as mol_scf
from pyscf.pbc import df, dft, gto
from pyscf.pbc.tools.k2gamma import k2gamma, to_supercell_ao_integrals

import decodense

# decimal tolerance
TOL = 5

# settings
PART = ('eda', 'atoms')

# init cell
cell = gto.Cell(verbose=0, output=None, basis='gth-szv-molopt-sr', pseudo='gth-pbe', atom='geom/h2o.xyz', a=4*np.eye(3), exp_to_discard=0.1)
cell.build()

# number of k-points / cell images in each direction
nkpt = 2
kmesh = [nkpt, nkpt, nkpt]
kpts = cell.make_kpts(kmesh)

# run k-point calculation
kmf = dft.KRKS(cell, kpts).density_fit(auxbasis="weigend")
kmf.xc = 'pbe,pbe'
edft = kmf.kernel()

# transform the k-point mf object to mf object for a supercell at gamma-point
mf_scf = k2gamma(kmf)
supcell, mo_coeff, mo_occ, e_mo = mf_scf.mol, mf_scf.mo_coeff, mf_scf.mo_occ, mf_scf.mo_energy
j_int = to_supercell_ao_integrals(cell, kpts, kmf.get_j())
mf = dft.RKS(supcell).density_fit().apply(mol_scf.addons.remove_linear_dep_)
mf.xc = 'pbe,pbe'
mf.with_df = df.df.DF(supcell)
mf.with_df.auxbasis = "weigend"
mf.mo_coeff, mf.mo_occ, mf.mo_energy = mo_coeff, mo_occ, e_mo
mf.initialize_grids(supcell, mf.make_rdm1(), supcell.make_kpts([1,1,1]))
mf.vj = j_int

def tearDownModule():
    global cell, supcell, kmf, mf
    cell.stdout.close()
    supcell.stdout.close()
    del cell, supcell, kmf, mf

class KnownValues(unittest.TestCase):
    def test(self):
        kmf_e_tot = (nkpt**3)*edft
        mf_e_tot = mf.energy_tot()
        for part in PART:
            with self.subTest(part=part):
                decomp = decodense.DecompCls(mo_basis='pm', pop_method='iao', mo_init='ibo', loc_exp=4, part=part)
                res = decodense.main(supcell, decomp, mf)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                self.assertAlmostEqual(kmf_e_tot, e_tot, TOL)
                self.assertAlmostEqual(mf_e_tot, e_tot, TOL)

if __name__ == '__main__':
    print('test: h2o_pbc_pbe_energy_gs')
    unittest.main()
