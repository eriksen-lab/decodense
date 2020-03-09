#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, scf, dft, lib
from mpi4py import MPI

from .system import DecompCls, sanity_check
from .orbitals import loc_orbs, reorder
from .properties import prop_tot
from .results import sort, info, table
from .tools import dim, time_str


def main(mol: gto.Mole, decomp: DecompCls):
        """
        main program
        """
        # sanity check
        sanity_check(decomp)

        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')
        # inter-atomic distance array
        rr = gto.mole.inter_distance(mol) * lib.param.BOHR

        # mf calculation
        if not decomp.dft:
            # hf calc
            print('\n\n ** hartree-fock')
            time = MPI.Wtime()
            mf = scf.RHF(mol)
            mf.conv_tol = 1.0e-12
            mf.kernel()
            assert mf.converged, 'HF not converged'
        else:
            # dft calc
            print('\n\n ** dft')
            time = MPI.Wtime()
            mf = dft.RKS(mol)
            mf.xc = decomp.xc
            mf.conv_tol = 1.0e-12
            mf.kernel()
            assert mf.converged, 'DFT not converged'

        # molecular dimensions
        mol.ncore, mol.nocc, mol.nvirt, mol.norb = dim(mol, mf.mo_occ)

        # print result header
        print(info(mol, decomp))

        # decompose property by means of canonical orbitals
        rep_idx, mo_can = np.arange(mol.nocc), mf.mo_coeff
        res_can = prop_tot(mol, mf, decomp.prop, mo_can[:, :mol.nocc], rep_idx)

        # decompose energy by means of localized MOs
        mo_loc = loc_orbs(mol, mf.mo_coeff, s, decomp.loc)
        rep_idx, centres = reorder(mol, s, mo_loc, decomp.pop, decomp.thres)
        res_loc = prop_tot(mol, mf, decomp.prop, mo_loc[:, :mol.nocc], rep_idx)

        # sort results
        res_can, res_loc, centres = sort(mol, res_can, res_loc, centres)

        # print results
        print(' done in: {:}'.format(time_str(MPI.Wtime() - time)))
        print(' ---------------\n')
        print(table(mol, decomp, res_can, res_loc, mf, centres, rr))


if __name__ == '__main__':
        # mol object
        mol = gto.Mole()
        mol.build(verbose = 0, output = None,
        basis = '631g', symmetry = True,
        atom = """O  0.  0.  0.1\n H -0.75  0. -0.48\n H  0.75  0. -0.48\n""")
        # decomp object
        decomp = DecompCls()
        # run decodense calculation
        main(mol, decomp)


