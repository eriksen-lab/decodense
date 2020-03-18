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
        # sanity check
        sanity_check(mol, decomp)

        # init time
        time = MPI.Wtime()

        # mf calculation
        if decomp.xc == '':
            # hf calc
            if mol.spin == 0:
                mf = scf.RHF(mol)
            else:
                if decomp.ref == 'restricted':
                    mf = scf.ROHF(mol)
                elif decomp.ref == 'unrestricted':
                    mf = scf.UHF(mol)
        else:
            # dft calc
            if mol.spin == 0:
                mf = dft.RKS(mol)
            else:
                if decomp.ref == 'restricted':
                    mf = dft.ROKS(mol)
                elif decomp.ref == 'unrestricted':
                    mf = dft.UKS(mol)
            mf.xc = decomp.xc
        mf.irrep_nelec = decomp.irrep_nelec
        mf.conv_tol = 1.e-12
        mf.kernel()
        assert mf.converged, 'mean-field calculation not converged'

        # reference property
        if decomp.prop == 'energy':
            decomp.prop_ref = mf.e_tot
        elif decomp.prop == 'dipole':
            decomp.prop_ref = scf.hf.dip_moment(mol, mf.make_rdm1(), unit='au', verbose=0)

        # restricted references
        if decomp.ref == 'restricted':
            mo = np.asarray((mf.mo_coeff,) * 2)
            mo_occ = np.asarray((np.zeros(mf.mo_occ.size, dtype=np.float64),) * 2)
            mo_occ[0][np.where(0. < mf.mo_occ)] += 1.
            mo_occ[1][np.where(1. < mf.mo_occ)] += 1.
        else:
            mo = mf.mo_coeff
            mo_occ = mf.mo_occ

        # nuclear property
        if decomp.prop == 'energy':
            decomp.prop_nuc = e_nuc(mol)
        elif decomp.prop == 'dipole':
            decomp.prop_nuc = dip_nuc(mol)

        # molecular dimensions
        mol.nalpha, mol.nbeta = dim(mol, mo_occ)
        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # compute localized molecular orbitals
        if decomp.orbs == 'localized':
            mo = loc_orbs(mol, mo, s, decomp.loc)

        # decompose electronic property
        rep_idx, cent = assign_rdm1s(mol, s, mo, mo_occ, decomp.pop, decomp.thres)
        decomp.prop_el = prop_tot(mol, mf, decomp.prop, mo, mo_occ, rep_idx)

        # collect electronic contributions in case of atom-based partitioning
        if decomp.part == 'atoms':
            decomp.prop_el = atom_part(mol, decomp.prop, decomp.prop_el, cent)

        # collect time
        decomp.time = MPI.Wtime() - time

        # print results
        if decomp.verbose:
            print(info(mol, decomp))
            if decomp.part == 'atoms':
                print(table_atoms(mol, decomp))
            elif decomp.part == 'bonds':
                print(table_bonds(mol, decomp, cent))

        return decomp.prop_el, decomp.prop_nuc


