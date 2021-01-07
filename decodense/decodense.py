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
from pyscf import lib, gto, scf, dft
from mpi4py import MPI
from typing import Dict, Tuple, List, Union, Any

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, assign_rdm1s
from .properties import prop_tot
from .tools import mf_calc, dim, make_rdm1, format_mf


def main(mol: gto.Mole, decomp: DecompCls, \
         dipole_origin: Union[List[float], np.ndarray] = [0.] * 3, \
         mf: Union[None, scf.hf.SCF, dft.rks.KohnShamDFT] = None) -> Dict[str, Any]:
        """
        main decodense program
        """
        # sanity check
        sanity_check(mol, decomp)

        # init time
        time = MPI.Wtime()

        # mf calculation
        if mf is None:
            mf, mo_coeff, mo_occ = mf_calc(mol, decomp.xc, decomp.ref, decomp.irrep_nelec, \
                                           decomp.conv_tol, decomp.grid_level, \
                                           decomp.verbose, decomp.mom)
        else:
            mo_coeff, mo_occ = format_mf(mf, decomp.ref)

        # molecular dimensions
        mol.alpha, mol.beta = dim(mol, mo_occ)
        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # compute localized molecular orbitals
        if decomp.loc != '':
            mo_coeff = loc_orbs(mol, mo_coeff, s, decomp.ref, decomp.loc)

        # inter-atomic distance array
        dist = gto.mole.inter_distance(mol) * lib.param.BOHR

        # decompose property
        if decomp.part in ['atoms', 'eda']:
            weights = assign_rdm1s(mol, s, mo_coeff, mo_occ, decomp.ref, decomp.pop, \
                                   decomp.part, decomp.verbose)[0]
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, \
                                  decomp.ref, decomp.prop, decomp.part, \
                                  decomp.cube, weights = weights, \
                                  dipole_origin = dipole_origin)
        elif decomp.part == 'bonds':
            rep_idx, centres = assign_rdm1s(mol, s, mo_coeff, mo_occ, decomp.ref, decomp.pop, \
                                                   decomp.part, decomp.verbose, thres = decomp.thres)
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, \
                                  decomp.ref, decomp.prop, decomp.part, \
                                  decomp.cube, rep_idx = rep_idx, \
                                  dipole_origin = dipole_origin)

        # determine spin
        decomp.res['ss'], decomp.res['s'] = scf.uhf.spin_square((mo_coeff[0][:, mol.alpha], \
                                                                 mo_coeff[1][:, mol.beta]), s)

        # collect time
        decomp.res['time'] = MPI.Wtime() - time

        # collect centres & dist
        if decomp.part == 'bonds':
            decomp.res['centres'] = centres
            decomp.res['dist'] = dist

        return decomp.res


