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
from pyscf import gto, scf
from mpi4py import MPI
from typing import Dict, Tuple, Any

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, assign_rdm1s
from .properties import prop_tot, e_nuc, dip_nuc
from .results import collect_res
from .tools import mf_calc, dim, make_rdm1


def main(mol: gto.Mole, decomp: DecompCls) -> Dict[str, Any]:
        """
        main decodense program
        """
        # sanity check
        sanity_check(mol, decomp)

        # init time
        time = MPI.Wtime()

        # mf calculation
        mf, mo_coeff, mo_occ = mf_calc(mol, decomp.xc, decomp.ref, decomp.irrep_nelec, \
                                       decomp.conv_tol, decomp.verbose, decomp.mom)

        # nuclear property
        if decomp.prop == 'energy':
            decomp.prop_nuc = e_nuc(mol)
        elif decomp.prop == 'dipole':
            decomp.prop_nuc = dip_nuc(mol)

        # molecular dimensions
        mol.alpha, mol.beta = dim(mol, mo_occ)
        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # compute localized molecular orbitals
        if decomp.loc != '':
            mo_coeff = loc_orbs(mol, mo_coeff, s, decomp.ref, decomp.loc)

        # determine spin
        decomp.ss, decomp.s = scf.uhf.spin_square((mo_coeff[0][:, mol.alpha], mo_coeff[1][:, mol.beta]), s)

        # decompose electronic property
        if decomp.part in ['atoms', 'eda']:
            weights = assign_rdm1s(mol, s, mo_coeff, mo_occ, decomp.ref, decomp.pop, \
                                   decomp.part, decomp.verbose)[0]
            decomp.prop_el = prop_tot(mol, mf, mo_coeff, mo_occ, decomp.ref, \
                                      decomp.prop, decomp.part, decomp.cube, weights = weights)
            decomp.pop_atom = np.sum(weights[0] + weights[1], axis=0)
        elif decomp.part == 'bonds':
            rep_idx, decomp.centres = assign_rdm1s(mol, s, mo_coeff, mo_occ, decomp.ref, decomp.pop, \
                                                   decomp.part, decomp.verbose, thres = decomp.thres)
            decomp.prop_el = prop_tot(mol, mf, mo_coeff, mo_occ, decomp.ref, \
                                      decomp.prop, decomp.part, decomp.cube, rep_idx = rep_idx)

        # collect time
        decomp.time = MPI.Wtime() - time

        return collect_res(decomp, mol)


