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
from .tools import dim, make_rdm1, format_mf, write_rdm1


def main(mol: gto.Mole, decomp: DecompCls, \
         mf: Union[None, scf.hf.SCF, dft.rks.KohnShamDFT], \
         loc_lst: Union[None, List[Any]] = None, \
         dipole_origin: Union[List[float], np.ndarray] = [0.] * 3) -> Dict[str, Any]:
        """
        main decodense program
        """
        # sanity check
        sanity_check(mol, decomp)

        # init time
        time = MPI.Wtime()

        # mf calculation
        mo_coeff, mo_occ = format_mf(mf, mol.spin)

        # molecular dimensions
        mol.alpha, mol.beta = dim(mol, mo_occ)
        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # compute localized molecular orbitals
        if decomp.loc != '':
            mo_coeff = loc_orbs(mol, mo_coeff, mo_occ, s, decomp.loc, loc_lst)

        # inter-atomic distance array
        dist = gto.mole.inter_distance(mol) * lib.param.BOHR

        # decompose property
        if decomp.part in ['atoms', 'eda']:
            weights = assign_rdm1s(mol, s, mo_coeff, mo_occ, decomp.pop, \
                                   decomp.part, decomp.multiproc, decomp.verbose)[0]
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, decomp.pop, \
                                  decomp.prop, decomp.part, decomp.multiproc, \
                                  weights = weights, dipole_origin = dipole_origin)
        elif decomp.part == 'bonds':
            rep_idx, centres = assign_rdm1s(mol, s, mo_coeff, mo_occ, decomp.pop, \
                                            decomp.part, decomp.multiproc, decomp.verbose, \
                                            thres = decomp.thres)
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, decomp.pop, \
                                  decomp.prop, decomp.part, decomp.multiproc, \
                                  rep_idx = rep_idx, dipole_origin = dipole_origin)

        # write rdm1s
        if decomp.write is not None:
            write_rdm1(mol, decomp.part, mo_coeff, mo_occ, decomp.write, \
                       weights if decomp.part == 'atoms' else None, \
                       rep_idx if decomp.part == 'bonds' else None)

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


