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
import warnings
import numpy as np
from pyscf import gto, scf, dft
from pyscf.pbc import gto as cgto
from pyscf.pbc import scf as cscf
from typing import Dict, Tuple, List, Union, Optional, Any

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, assign_rdm1s
from .properties import prop_tot
from .tools import make_natorb, mf_info, write_rdm1


def main(mol: Union[gto.Mole, cgto.Cell], decomp: DecompCls, \
         mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, cscf.RHF], \
         rdm1_orb: np.ndarray = None, \
         rdm1_eff: np.ndarray = None, \
         loc_lst: Optional[Any] = None) -> Dict[str, Any]:
        """
        main decodense program
        """
        # sanity check
        sanity_check(mol, decomp)
    
        if isinstance(mol, cgto.Cell) and (decomp.prop == 'energy'):
            if decomp.part == 'atoms':
                decomp.res = prop_tot(mol, mf, (None, None), (None, None), (None, None), decomp.pop, \
                                      decomp.prop, decomp.part, decomp.ndo, decomp.multiproc, decomp.gauge_origin, \
                                      weights = None)

                warnings.warn('PBC module is in development, but atomwise nuclear-nuclear repulsion' + \
                         ' term of energy at gamma point has been computed', Warning)
                return decomp.res
            else:
                sys.exit('PBC module is in development, can only compute atomwise ' + \
                         'nuclear-nuclear repulsion term of energy at gamma point')
        elif isinstance(mol, cgto.Cell) and (decomp.prop != 'energy'):
            sys.exit('PBC module is in development, the only valid choice for property: energy')

        # format orbitals from mean-field calculation
        if rdm1_orb is None:
            mo_coeff, mo_occ = mf_info(mf)
        else:
            mo_coeff, mo_occ = make_natorb(mol, np.asarray(mf.mo_coeff), np.asarray(rdm1_orb))

        # compute localized molecular orbitals
        if decomp.loc != '':
            mo_coeff = loc_orbs(mol, mo_coeff, mo_occ, decomp.loc, decomp.ndo, loc_lst)

        # decompose property
        if decomp.part in ['atoms', 'eda']:
            # compute population weights
            weights = assign_rdm1s(mol, mo_coeff, mo_occ, decomp.pop, decomp.part, \
                                   decomp.ndo, decomp.multiproc, decomp.verbose)
            # compute decomposed results
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, rdm1_eff, \
                                  decomp.pop, decomp.prop, decomp.part, \
                                  decomp.ndo, decomp.multiproc, decomp.gauge_origin, \
                                  weights = weights)
        else: # orbs
            # compute decomposed results
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, rdm1_eff, \
                                  decomp.pop, decomp.prop, decomp.part, \
                                  decomp.ndo, decomp.multiproc, decomp.gauge_origin)

        # write rdm1s
        if decomp.write != '':
            write_rdm1(mol, decomp.part, mo_coeff, mo_occ, decomp.write, weights)

        return decomp.res

