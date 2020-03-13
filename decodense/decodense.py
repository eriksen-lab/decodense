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
from pyscf import gto, scf, dft
from mpi4py import MPI
from typing import Tuple

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, assign_rdm1s, atom_part
from .properties import prop_tot, e_nuc, dip_nuc
from .results import info, table_atoms, table_bonds
from .tools import dim


def main(mol: gto.Mole, decomp: DecompCls) -> Tuple[np.ndarray, np.ndarray]:
        """
        main decodense program
        """
        # init time
        time = MPI.Wtime()

        # sanity check
        sanity_check(decomp)

        # mf calculation
        if decomp.xc == '':
            # hf calc
            mf = scf.RHF(mol)
            mf.conv_tol = 1.0e-12
            mf.kernel()
            assert mf.converged, 'HF not converged'
        else:
            # dft calc
            mf = dft.RKS(mol)
            mf.xc = decomp.xc
            mf.conv_tol = 1.0e-12
            mf.kernel()
            assert mf.converged, 'DFT not converged'

        # reference property
        if decomp.prop == 'energy':
            decomp.prop_ref = mf.e_tot
        elif decomp.prop == 'dipole':
            decomp.prop_ref = scf.hf.dip_moment(mol, mf.make_rdm1(), unit='au', verbose=0)

        # molecular dimensions
        mol.ncore, mol.nocc, mol.nvirt, mol.norb = dim(mol, mf.mo_occ)
        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # decompose property by means of canonical orbitals
        mo_can = mf.mo_coeff
        rep_idx, cent_can = assign_rdm1s(mol, s, mo_can, decomp.pop, decomp.thres)
        res_can = prop_tot(mol, mf, decomp.prop, mo_can[:, :mol.nocc], rep_idx)

        # decompose energy by means of localized MOs
        mo_loc = loc_orbs(mol, mf.mo_coeff, s, decomp.loc)
        rep_idx, cent_loc = assign_rdm1s(mol, s, mo_loc, decomp.pop, decomp.thres)
        res_loc = prop_tot(mol, mf, decomp.prop, mo_loc[:, :mol.nocc], rep_idx)

        # collect contributions in case of atom-based partitioning
        if decomp.part == 'atoms':
            res_can = atom_part(mol, decomp.prop, res_can, cent_can)
            res_loc = atom_part(mol, decomp.prop, res_loc, cent_loc)
            # add nuclear contributions
            if decomp.prop == 'energy':
                res_can += e_nuc(mol)
                res_loc += e_nuc(mol)
            elif decomp.prop == 'dipole':
                res_can += dip_nuc(mol)
                res_loc += dip_nuc(mol)

        # collect time
        decomp.time = MPI.Wtime() - time

        # print results
        if decomp.verbose:
            print(info(mol, decomp))
            if decomp.part == 'atoms':
                print(table_atoms(mol, decomp, res_can, 'canonical'))
                print(table_atoms(mol, decomp, res_loc, 'localized'))
            elif decomp.part == 'bonds':
                print(table_bonds(mol, decomp, res_can, cent_can, 'canonical'))
                print(table_bonds(mol, decomp, res_loc, cent_loc, 'localized'))

        return res_can, res_loc


