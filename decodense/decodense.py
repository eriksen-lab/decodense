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
         mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
         loc_lst: Union[None, List[Any]] = None) -> Dict[str, Any]:
        """
        main decodense program
        """
        # sanity check
        sanity_check(mol, decomp)

        # init time
        time = MPI.Wtime()

        # mf calculation
        mo_coeff, mo_occ = format_mf(mf)

        # compute localized molecular orbitals
        if decomp.loc != '':
            mo_coeff = loc_orbs(mol, mo_coeff, mo_occ, decomp.loc, loc_lst)

        # decompose property
        if decomp.part in ['atoms', 'eda']:
            # compute population weights
            weights = assign_rdm1s(mol, mo_coeff, mo_occ, decomp.pop, decomp.part, \
                                   multiproc = decomp.multiproc, verbose = decomp.verbose)[0]
            # compute decomposed results
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, decomp.pop, \
                                  decomp.prop, decomp.part, decomp.multiproc, \
                                  decomp.gauge_origin, weights = weights)
        elif decomp.part == 'bonds':
            # compute repetitive indices & centres
            rep_idx, centres = assign_rdm1s(mol, mo_coeff, mo_occ, decomp.pop, decomp.part, \
                                            multiproc = decomp.multiproc, verbose = decomp.verbose, \
                                            thres = decomp.thres)
            # compute decomposed results
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, decomp.pop, \
                                  decomp.prop, decomp.part, decomp.multiproc, \
                                  decomp.gauge_origin, rep_idx = rep_idx)
            # save centres & inter-atomic distances
            decomp.res['centres'] = centres
            decomp.res['dist'] = gto.mole.inter_distance(mol) * lib.param.BOHR

        # write rdm1s
        if decomp.write != '':
            write_rdm1(mol, decomp.part, mo_coeff, mo_occ, decomp.write, \
                       weights if decomp.part == 'atoms' else None, \
                       rep_idx if decomp.part == 'bonds' else None)

        # collect time
        decomp.res['time'] = MPI.Wtime() - time

        return decomp.res


