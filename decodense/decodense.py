#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program
"""

__author__ = 'Janus Juul Eriksen, Technical University of Denmark, DK'
__maintainer__ = 'Janus Juul Eriksen'
__email__ = 'janus@kemi.dtu.dk'
__status__ = 'Development'

import numpy as np
import pandas as pd
from pyscf import gto, scf, dft
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from typing import Dict, Tuple, List, Union, Optional, Any

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, assign_rdm1s
from .properties import prop_tot
from .tools import make_natorb, mf_info, write_rdm1
from .results import fmt

def main(mol: Union[gto.Mole, pbc_gto.Cell], decomp: DecompCls, \
         mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.hf.RHF, pbc_dft.rks.RKS], \
         rdm1_orb: np.ndarray = None, \
         rdm1_eff: np.ndarray = None) -> pd.DataFrame:
        """
        main decodense program
        """
        # sanity check
        sanity_check(mol, mf, decomp)

        # format orbitals from mean-field calculation
        if rdm1_orb is None:
            mo_coeff, mo_occ = mf_info(mf)
        else:
            mo_coeff, mo_occ = make_natorb(mol, np.asarray(mf.mo_coeff), np.asarray(rdm1_orb))

        # compute localized MOs
        if decomp.mo_basis != 'can':
            mo_coeff = loc_orbs(mol, mf, mo_coeff, mo_occ, \
                                decomp.mo_basis, decomp.pop_method, decomp.mo_init, decomp.loc_exp, \
                                decomp.ndo, decomp.verbose)

        # decompose property
        if decomp.part in ['atoms', 'eda']:
            # compute population weights
            weights = assign_rdm1s(mol, mf, mo_coeff, mo_occ, decomp.pop_method, decomp.part, \
                                   decomp.ndo, decomp.verbose)
            # compute decomposed results
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, rdm1_eff, \
                                  decomp.pop_method, decomp.prop, decomp.part, \
                                  decomp.ndo, decomp.gauge_origin, \
                                  weights = weights)
        else: # orbs
            # compute decomposed results
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, rdm1_eff, \
                                  decomp.pop_method, decomp.prop, decomp.part, \
                                  decomp.ndo, decomp.gauge_origin)

        # write rdm1s
        if decomp.write != '':
            write_rdm1(mol, decomp.part, mo_coeff, mo_occ, decomp.write, weights)

        return fmt(mol, decomp.res, decomp.unit, decomp.ndo)

