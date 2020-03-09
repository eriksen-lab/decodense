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
from typing import Tuple

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, reorder
from .properties import prop_tot
from .results import sort, info, table
from .tools import dim


def main(mol: gto.Mole, decomp: DecompCls) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        main decodense program
        """
        # sanity check
        sanity_check(decomp)

        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')
        # inter-atomic distance array
        dist = gto.mole.inter_distance(mol) * lib.param.BOHR

        # mf calculation
        if decomp.xc == '':
            # hf calc
            time = MPI.Wtime()
            mf = scf.RHF(mol)
            mf.conv_tol = 1.0e-12
            mf.kernel()
            assert mf.converged, 'HF not converged'
        else:
            # dft calc
            time = MPI.Wtime()
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

        # print result header
        if decomp.verbose:
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

        # collect time
        decomp.time = MPI.Wtime() - time

        # print results
        if decomp.verbose:
            print(table(mol, decomp, res_can, res_loc, mf, centres, dist))

        return res_can, res_loc, centres, dist


