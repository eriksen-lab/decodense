#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
orbitals module containing all functions related to orbital transformations and assignments in mf_decomp
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import lo


def loc_orbs(mol, mo_coeff, s, variant):
    """
    this function returns a set of localized MOs of a specific variant

    :param mol: pyscf mol object
    :param mo_coeff: initial mo coefficients. numpy array of shape (n_orb, n_orb)
    :param s: overlap matrix. numpy array of shape (n_orb, n_orb)
    :param variant: localization variant. string
    :return: numpy array of shape (n_orb, n_orb)
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

    elif variant == 'ibo':

        # IAOs
        iao_core = lo.iao.iao(mol, mo_coeff[:, :mol.ncore])
        iao_val = lo.iao.iao(mol, mo_coeff[:, mol.ncore:mol.nocc])

        # orthogonalize IAOs
        iao_core = lo.vec_lowdin(iao_core, s)
        iao_val = lo.vec_lowdin(iao_val, s)

        # IBOs
        mo_coeff[:, :mol.ncore] = lo.ibo.ibo(mol, mo_coeff[:, :mol.ncore], iao_core, verbose=0)
        mo_coeff[:, mol.ncore:mol.nocc] = lo.ibo.ibo(mol, mo_coeff[:, mol.ncore:mol.nocc], iao_val, verbose=0)

    else:

        raise RuntimeError('\n invalid localization procedure. valid choices: `pm` and `ibo`\n')

    return mo_coeff


def set_ncore(mol):
    """
    this function returns number of core orbitals

    :param mol: pyscf mol object
    :return: integer
    """
    # init ncore
    ncore = 0

    # loop over atoms
    for i in range(mol.natm):

        if mol.atom_charge(i) > 2: ncore += 1
        if mol.atom_charge(i) > 12: ncore += 4
        if mol.atom_charge(i) > 20: ncore += 4
        if mol.atom_charge(i) > 30: ncore += 6

    return ncore


def charge_centres(mol, s, rdm1):
    """
    this function returns the mulliken charges on the individual atoms

    :param mol: pyscf mol object
    :param s: overlap matrix. numpy array of shape (n_orb, n_orb)
    :param rdm1: orbital specific rdm1. numpy array of shape (n_orb, n_orb)
    :return: numpy array of shape (natm,)
    """
    # mulliken population matrix
    pop = np.einsum('ij,ji->i', rdm1, s).real

    # init charges
    charges = np.zeros(mol.natm)

    # loop over AOs
    for i, s in enumerate(mol.ao_labels(fmt=None)):

        charges[s[0]] += pop[i]

    # get sorted indices
    max_idx = np.argsort(charges)[::-1]

    if np.abs(charges[max_idx[0]]) / np.abs((charges[max_idx[0]] + charges[max_idx[1]])) > 0.95:

        # core orbital
        return np.sort(np.array([max_idx[0], max_idx[0]], dtype=np.int))

    else:

        # valence orbitals
        return np.sort(np.array([max_idx[0], max_idx[1]], dtype=np.int))


