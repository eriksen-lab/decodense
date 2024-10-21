#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program
"""

__author__ = "Janus Juul Eriksen, Technical University of Denmark, DK"
__maintainer__ = "Janus Juul Eriksen"
__email__ = "janus@kemi.dtu.dk"
__status__ = "Development"

import numpy as np
import pandas as pd
from pyscf import gto, scf, dft
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from typing import Union, Optional, Tuple

from .decomp import DecompCls, sanity_check
from .orbitals import assign_rdm1s
from .properties import prop_tot
from .tools import write_rdm1
from .results import fmt


def main(
    mol: Union[gto.Mole, pbc_gto.Cell],
    decomp: DecompCls,
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.hf.RHF, pbc_dft.rks.RKS],
    mo_coeff: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    mo_occ: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
    rdm1: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    main decodense program
    """
    # sanity check
    sanity_check(mol, mf, decomp, mo_coeff, mo_occ)

    # ensure mo coefficients are in the correct shape
    if isinstance(mo_coeff, np.ndarray):
        if mo_coeff.ndim == 2:
            mo_coeff = 2 * (mo_coeff,)
        else:
            mo_coeff = tuple(mo_coeff)

    # assume occupation if not provided
    if mo_occ is None:
        mo_occ = (np.ones(mo_coeff[0].shape[1]), np.ones(mo_coeff[1].shape[1]))
    elif isinstance(mo_occ, np.ndarray):
        if mo_occ.ndim == 1:
            mo_occ = 2 * (mo_occ / 2,)
        else:
            mo_occ = tuple(mo_occ)

    # compute population weights
    weights = assign_rdm1s(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        decomp.minao,
        decomp.pop_method,
        decomp.ndo,
        decomp.verbose,
    )

    # compute decomposed results
    decomp.res = prop_tot(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        rdm1,
        decomp.minao,
        decomp.pop_method,
        decomp.prop,
        decomp.part,
        decomp.ndo,
        decomp.gauge_origin,
        weights,
    )

    # write rdm1s
    if decomp.write != "":
        write_rdm1(mol, decomp.part, mo_coeff, mo_occ, decomp.write, weights)

    return fmt(mol, decomp.res, decomp.unit, decomp.ndo)
