#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
orbitals module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, scf, dft, lo
from typing import List, Tuple, Union


def loc_orbs(mol: gto.Mole, mo_coeff: np.ndarray, s: np.ndarray, variant: str) -> np.ndarray:
    """
    this function returns a set of localized MOs of a specific variant
    """
    # init localizer
    if variant == 'pm':
        # pipek-mezey procedure
        loc_core = lo.PM(mol, mo_coeff[:, :mol.ncore])
        loc_val = lo.PM(mol, mo_coeff[:, mol.ncore:mol.nocc])
        # convergence threshold
        loc_core.conv_tol = loc_val.conv_tol = 1.0e-10
        # localize core and valence occupied orbitals
        mo_coeff[:, :mol.ncore] = loc_core.kernel()
        mo_coeff[:, mol.ncore:mol.nocc] = loc_val.kernel()
    elif 'ibo' in variant:
        # IAOs
        iao_core = lo.iao.iao(mol, mo_coeff[:, :mol.ncore])
        iao_val = lo.iao.iao(mol, mo_coeff[:, mol.ncore:mol.nocc])
        # orthogonalize IAOs
        iao_core = lo.vec_lowdin(iao_core, s)
        iao_val = lo.vec_lowdin(iao_val, s)
        # IBOs
        if variant == 'ibo-2':
            mo_coeff[:, :mol.ncore] = lo.ibo.ibo(mol, mo_coeff[:, :mol.ncore], \
                                                    iaos=iao_core, exponent=2, verbose=0)
            mo_coeff[:, mol.ncore:mol.nocc] = lo.ibo.ibo(mol, mo_coeff[:, mol.ncore:mol.nocc], \
                                                            iaos=iao_val, exponent=2, verbose=0)
        elif variant == 'ibo-4':
            mo_coeff[:, :mol.ncore] = lo.ibo.ibo(mol, mo_coeff[:, :mol.ncore], \
                                                    iaos=iao_core, exponent=4, verbose=0)
            mo_coeff[:, mol.ncore:mol.nocc] = lo.ibo.ibo(mol, mo_coeff[:, mol.ncore:mol.nocc], \
                                                            iaos=iao_val, exponent=4, verbose=0)
        else:
            raise RuntimeError('\n invalid localization procedure. '
                               'valid choices: `pm`, `ibo-2`, and `ibo-4`\n')
    else:
        raise RuntimeError('\n invalid localization procedure. '
                           'valid choices: `pm`, `ibo-2`, and `ibo-4`\n')

    return mo_coeff


def reorder(mol: gto.Mole, s: np.ndarray, mo_coeff: np.ndarray, \
                pop: str, thres: float) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    this function returns a list of repetitive center indices and an array of unique charge centres
    """
    # init charge_centres array
    centres = np.zeros([mol.nocc, 2], dtype=np.int)

    for i in range(mol.nocc):
        # get orbital
        orb = mo_coeff[:, i].reshape(mol.norb, 1)
        # orbital-specific rdm1
        rdm1_orb = np.einsum('ip,jp->ij', orb, orb) * 2.
        # charge centres of orbital
        centres[i] = _charge_centres(mol, s, orb, rdm1_orb, pop, thres)

    # search for the unique centres for local results
    centres_unique = np.unique(centres, axis=0)
    # repetitive centres
    rep_idx = [np.where((centres == i).all(axis=1))[0] for i in centres_unique]

    return rep_idx, centres_unique


def _charge_centres(mol: gto.Mole, s: np.ndarray, orb: np.ndarray, \
                        rdm1: np.ndarray, pop: str, thres: float) -> np.ndarray:
    """
    this function returns a single atom/pair of atoms onto which a given MO is assigned
    """
    if pop == 'mulliken':

        # traditional mulliken charges
        charges = _mulliken_charges(mol, s, rdm1)

    elif pop == 'iao':

        # base mulliken charges on IAOs
        iao = lo.iao.iao(mol, orb)
        iao = lo.vec_lowdin(iao, s)
        orb_iao = np.einsum('ki,kl,lj', iao, s, orb)
        rdm1_iao = np.einsum('ip,jp->ij', orb_iao, orb_iao) * 2.
        pmol = mol.copy()
        pmol.build(False, False, basis='minao')

        charges = _mulliken_charges(pmol, np.eye(pmol.nao_nr()), rdm1_iao)

    else:

        raise RuntimeError('\n invalid population scheme. '
                           'valid choices: `mulliken` and `iao`\n')

    # get sorted indices
    max_idx = np.argsort(charges)[::-1]

    if np.abs(charges[max_idx[0]]) / np.abs((charges[max_idx[0]] + charges[max_idx[1]])) > thres:
        # core orbital
        return np.sort(np.array([max_idx[0], max_idx[0]], dtype=np.int))
    else:
        # valence orbitals
        return np.sort(np.array([max_idx[0], max_idx[1]], dtype=np.int))


def _mulliken_charges(mol: gto.Mole, s: np.ndarray, rdm1: np.ndarray) -> np.ndarray:
    """
    this function returns the mulliken charges on the individual atoms
    """
    # mulliken population matrix
    pop = np.einsum('ij,ji->i', rdm1, s).real
    # init charges
    charges = np.zeros(mol.natm)

    # loop over AOs
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        charges[s[0]] += pop[i]

    return charges


