#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
energy module containing all functions related to energy calculations in mf_decomp
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import scf

import orbitals


def e_elec(h_core, vj, vk, rdm1):
    """
    this function returns a contribution to a mean-field energy from rdm1:
    E(rdm1) = 2. * Tr[h * rdm1] + Tr[v_eff(rdm1_tot) * rdm1]

    :param h_core: core hamiltonian. numpy array of shape (n_orb, n_orb)
    :param vj: coulumb potential. numpy array of shape (n_orb, n_orb)
    :param vk: exchange potential. numpy array of shape (n_orb, n_orb)
    :param rdm1: orbital specific rdm1. numpy array of shape (n_orb, n_orb)
    :return: scalar
    """
    # contribution from core hamiltonian
    e_core = np.einsum('ij,ji', h_core, rdm1)

    # contribution from effective potential
    e_veff = np.einsum('ij,ji', vj - vk * .5, rdm1) * .5

    return e_core + e_veff


def e_tot(mol, s, mo_coeff, alpha=1.):
    """
    this function returns a sorted orbital-decomposed mean-field energy for a given orbital variant

    :param mol: pyscf mol object
    :param mf: pyscf mean-field object
    :param s: overlap matrix. numpy array of shape (n_orb, n_orb)
    :param mo_coeff: mo coefficients. numpy array of shape (n_orb, n_orb)
    :param alpha. exact exchange ratio for hf and hybrid xc functionals. scalar
    :return: numpy array of shape (nocc,)
    """
    # compute total 1-RDM (AO basis)
    rdm1 = np.einsum('ip,jp->ij', mo_coeff[:, :mol.nocc], mo_coeff[:, :mol.nocc]) * 2.

    # core hamiltonian
    h_core = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

    # fock potential
    vj, vk = scf.hf.get_jk(mol, rdm1)
    # scale amount of exact exchange for dft
    vk *= alpha

    # init orbital energy array
    e_orb = np.zeros(mol.nocc, dtype=np.float64)
    # init charge_centres array
    centres = np.zeros([mol.nocc, 2], dtype=np.int)

    # loop over orbitals
    for orb in range(mol.nocc):

        # orbital-specific 1rdm
        rdm1_orb = np.einsum('ip,jp->ij', mo_coeff[:, [orb]], mo_coeff[:, [orb]]) * 2.

        # charge centres of orbital
        centres[orb] = orbitals.charge_centres(mol, s, rdm1_orb)

        # energy from individual orbitals
        e_orb[orb] = e_elec(h_core, vj, vk, rdm1_orb)

    return e_orb, centres


