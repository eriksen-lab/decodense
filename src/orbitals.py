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


def loc_orbs(mol, mf, s, variant):
    """
    this function returns a set of localized MOs of a specific variant

    :param mol: pyscf mol object
    :param mf: pyscf mf object
    :param s: overlap matrix. numpy array of shape (n_orb, n_orb)
    :param variant: localization variant. string
    :return: numpy array of shape (n_orb, n_orb)
    """
    # copy MOs from mean-field object
    mo_coeff = np.copy(mf.mo_coeff)

    # init localizer
    if variant == 'boys':

        # Foster-Boys procedure
        loc = lo.Boys(mol, mo_coeff[:, :mol.nocc])

    elif variant == 'pm':

        # Pipek-Mezey procedure
        loc = lo.PM(mol, mo_coeff[:, :mol.nocc])

    elif variant == 'er':

        # Edmiston-Ruedenberg procedure
        loc = lo.ER(mol, mo_coeff[:, :mol.nocc])

    elif variant == 'ibo':

        # compute IAOs
        a = lo.iao.iao(mol, mo_coeff[:, :mol.nocc])

        # orthogonalize IAOs
        a = lo.vec_lowdin(a, s)

        # IBOs via Pipek-Mezey procedure
        loc = lo.ibo.PM(mol, mo_coeff[:, :mol.nocc], a)

    else:

        raise RuntimeError('\n unknown localization procedure. valid choices: `boys`, `pm`, `er`, and `ibo`\n')

    # convergence threshold
    loc.conv_tol = 1.0e-10

    # localize occupied orbitals
    mo_coeff[:, :mol.nocc] = loc.kernel()

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
        return np.array([max_idx[0], max_idx[0]], dtype=np.int)

    else:

        # valence orbitals
        return np.array([max_idx[0], max_idx[1]], dtype=np.int)


