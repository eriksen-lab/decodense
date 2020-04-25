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
from typing import Dict, Tuple, Any

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, assign_rdm1s, partition
from .properties import prop_tot, e_nuc, dip_nuc
from .results import collect_res
from .tools import dim


def main(mol: gto.Mole, decomp: DecompCls) -> Dict[str, Any]:
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
        mf.verbose = decomp.verbose
        mf.conv_tol = decomp.conv_tol
        mf.kernel()
        assert mf.converged, 'mean-field calculation not converged'

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
        mol.ncore, mol.nalpha, mol.nbeta = dim(mol, mo_occ)
        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # compute localized molecular orbitals
        if decomp.loc != '':
            mo = loc_orbs(mol, mo, s, decomp.loc)

        # determine spin
        decomp.ss, decomp.s = scf.uhf.spin_square((mo[0][:, :mol.nalpha], mo[1][:, :mol.nbeta]), s)

        # decompose electronic property
        weights = assign_rdm1s(mol, mf, s, mo, mo_occ, decomp.pop)
        decomp.prop_el = prop_tot(mol, mf, decomp.prop, mo, mo_occ)

        # collect electronic contributions
        decomp.prop_el = partition(mol, decomp.prop, decomp.prop_el, weights)
        decomp.prop_tot = decomp.prop_el + decomp.prop_nuc

        # collect time
        decomp.time = MPI.Wtime() - time

        return collect_res(decomp, mol)


