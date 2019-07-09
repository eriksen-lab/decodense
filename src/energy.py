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
from pyscf import tools as pyscf_tools

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


def e_tot(mol, system, orb_type, ao_dip, mo_coeff, rep_idx, alpha=1.):
    """
    this function returns a sorted orbital-decomposed mean-field energy for a given orbital variant

    :param mol: pyscf mol object
    :param system: system information. dict
    :param orb_type: type of decomposition. string
    :param ao_dipole: dipole integrals in ao basis. numpy array of shape (3, n_basis, n_basis)
    :param mo_coeff: mo coefficients. numpy array of shape (n_basis, n_unique)
    :param rep_idx: list of repetitive indices. list of numpy arrays of various shapes
    :param alpha. exact exchange ratio for hf and hybrid xc functionals. scalar
    :return: numpy array of shape (n_unique,) [e_orb],
             numpy array of shape (n_unique, 3) [dip_orb],
             numpy array of shape (n_unique, 2) [centres]
    """
    # compute total 1-RDM (AO basis)
    rdm1 = np.einsum('ip,jp->ij', mo_coeff, mo_coeff) * 2.

    # core hamiltonian
    h_core = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

    # fock potential
    vj, vk = scf.hf.get_jk(mol, rdm1)
    # scale amount of exact exchange for dft
    vk *= alpha

    # init orbital-specific energy array
    e_orb = np.zeros(len(rep_idx), dtype=np.float64)
    # init orbital-specific dipole array
    dip_orb = np.zeros([len(rep_idx), 3], dtype=np.float64)

    # loop over orbitals
    for i, j in enumerate(rep_idx):

        # get orbital
        orb = mo_coeff[:, j].reshape(mo_coeff.shape[0], -1)

        # orbital-specific rdm1
        rdm1_orb = np.einsum('ip,jp->ij', orb, orb) * 2.

        # write cube file
        if system['cube']:
            out_path = system['out_{:}_path'.format(orb_type)]
            pyscf_tools.cubegen.density(mol, out_path + '/rdm1_{:}_{:}_tmp.cube'.format(orb_type, i), rdm1_orb)

        # energy from individual orbitals
        e_orb[i] = e_elec(h_core, vj, vk, rdm1_orb)

        # dipole from individual orbitals
        dip_orb[i] = np.einsum('xij,ji->x', ao_dip, rdm1_orb).real

    return e_orb, dip_orb


