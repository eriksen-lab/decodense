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


def e_tot(mol, mf, s, ao_dip, mo_coeff, pop='mulliken', alpha=1.):
    """
    this function returns a sorted orbital-decomposed mean-field energy for a given orbital variant

    :param mol: pyscf mol object
    :param mf: pyscf mean-field object
    :param s: overlap matrix. numpy array of shape (n_orb, n_orb)
    :param ao_dipole: dipole integrals in ao basis. numpy array of shape (3, n_orb, n_orb)
    :param mo_coeff: mo coefficients. numpy array of shape (n_orb, n_orb)
    :param pop: population scheme. string
    :param alpha. exact exchange ratio for hf and hybrid xc functionals. scalar
    :return: numpy array of shape (nocc,) [e_orb],
             numpy array of shape (nocc, 3) [dip_orb],
             numpy array of shape (nocc, 2) [centres]
    """
    # compute total 1-RDM (AO basis)
    rdm1 = np.einsum('ip,jp->ij', mo_coeff[:, :mol.nocc], mo_coeff[:, :mol.nocc]) * 2.

    # core hamiltonian
    h_core = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

    # fock potential
    vj, vk = scf.hf.get_jk(mol, rdm1)
    # scale amount of exact exchange for dft
    vk *= alpha

    # init orbital-specific energy array
    e_orb = np.zeros(mol.nocc, dtype=np.float64)
    # init orbital-specific dipole array
    dip_orb = np.zeros([mol.nocc, 3], dtype=np.float64)
    # init charge_centres array
    centres = np.zeros([mol.nocc, 2], dtype=np.int)

    # loop over orbitals
    for i in range(mol.nocc):

        # get orbital
        orb = mo_coeff[:, i].reshape(mol.norb, 1)

        # orbital-specific rdm1
        rdm1_orb = np.einsum('ip,jp->ij', orb, orb) * 2.

        # charge centres of orbital
        centres[i] = orbitals.charge_centres(mol, mf, s, orb, rdm1_orb, pop)

        # energy from individual orbitals
        e_orb[i] = e_elec(h_core, vj, vk, rdm1_orb)

        # dipole from individual orbitals
        dip_orb[i] = np.einsum('xij,ji->x', ao_dip, rdm1_orb).real

    return e_orb, dip_orb, centres


