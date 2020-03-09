#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import os.path
import shutil
import numpy as np
from pyscf import gto, scf, dft, lib
from mpi4py import MPI

import system
import orbitals
import energy
import results
import tools


def main(mol: gto.Mole, decomp: system.DecompCls):
        """
        main program
        """
        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')
        # inter-atomic distance array
        rr = gto.mole.inter_distance(mol) * lib.param.BOHR
        # nuclear repulsion energy and dipole moment
        if decomp.prop == 'energy':
            e_nuc = mol.energy_nuc()
        elif decomp.prop == 'dipole':
            dip_nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())

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
        mol.ncore, mol.nocc, mol.nvirt, mol.norb = tools.dim(mol, mf)

        # print result header
        print(results.main(mol, decomp))

        # decompose property by means of canonical orbitals
        rep_idx, mo_can = np.arange(mol.nocc), mf.mo_coeff
        res_can = energy.e_tot(mol, mf, decomp.prop, mo_can[:, :mol.nocc], rep_idx)

        # decompose energy by means of localized MOs
        mo_loc = orbitals.loc_orbs(mol, mf.mo_coeff, s, decomp.loc)
        rep_idx, centres = orbitals.reorder(mol, s, mo_loc, decomp.pop, decomp.thres)
        res_loc = energy.e_tot(mol, mf, decomp.prop, mo_loc[:, :mol.nocc], rep_idx)

        # sort results
        res_can, res_loc, centres = results.sort(mol, res_can, res_loc, centres)

        # print results
        print(' done in: {:}'.format(tools.time_str(MPI.Wtime() - time)))
        print(' ---------------\n')
        if decomp.prop == 'energy':
            print(results.energy(mol, res_can, res_loc, e_nuc, mf.e_tot, centres, rr))
        elif decomp.prop == 'dipole':
            print(results.dipole(mol, res_can, res_loc, dip_nuc, \
                                 scf.hf.dip_moment(mol, mf.make_rdm1(), unit='au', verbose=0), centres, rr))


if __name__ == '__main__':
        # mol object
        mol = gto.Mole()
        mol.build(verbose = 0, output = None,
        basis = '631g', symmetry = True,
        atom = """O  0.  0.  0.1\n H -0.75  0. -0.48\n H  0.75  0. -0.48\n""")
        # decomp object
        decomp = system.DecompCls()
        # run decodense calculation
        main(mol, decomp)


