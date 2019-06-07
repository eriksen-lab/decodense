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
    e_core = np.einsum('ij,ji', h_core, rdm1) * 2.

    # contribution from effective potential
    e_veff = np.einsum('ij,ji', vj, rdm1)
    if vk is not None:
        e_veff -= np.einsum('ij,ji', vk * .5, rdm1)

    return e_core + e_veff


def e_tot(mol, mf, s, mo_coeff, dft=False):
    """
    this function returns a sorted orbital-decomposed mean-field energy for a given orbital variant

    :param mol: pyscf mol object
    :param mf: pyscf mean-field object
    :param s: overlap matrix. numpy array of shape (n_orb, n_orb)
    :param mo_coeff: mo coefficients. numpy array of shape (n_orb, n_orb)
    :param dft: dft logical. bool
    :return: numpy array of shape (nocc,)
    """
    # compute 1-RDM (in AO representation)
    rdm1 = mf.make_rdm1(mo_coeff, mf.mo_occ)

    # core hamiltonian
    h_core = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

    # mean-field effective potential
    if dft:

        v_dft = mf.get_veff(mol, rdm1)
        vj, vk = v_dft.vj, v_dft.vk

    else:

        vj, vk = mf.get_jk(mol, rdm1)

    # init orbital energy array
    e_orb = np.zeros(mol.nocc, dtype=np.float64)
    # init charge_centres array
    centres = np.zeros([mol.nocc, 2], dtype=np.int)

    # loop over orbitals
    for orb in range(mol.nocc):

        # orbital-specific 1rdm
        rdm1_orb = np.einsum('ip,jp->ij', mo_coeff[:, [orb]], mo_coeff[:, [orb]])

        # charge centres of orbital
        centres[orb] = orbitals.charge_centres(mol, s, rdm1_orb)

        # energy from individual orbitals
        e_orb[orb] = e_elec(h_core, vj, vk, rdm1_orb)

    # convert centres to array
    centres = np.array(centres)

    # sort arrays wrt e_orb
    centres = centres[np.argsort(e_orb)]
    e_orb = np.sort(e_orb)

    return e_orb, centres


def energy_nuc(mol):
    """
    this function returns the nuclear repulsion energy for all atoms of the system
    see: pyscf/gto/mole.py

    :param mol: pyscf mol object
    :return: numpy array of shape (natm,)
    """
    # charges
    charges = mol.atom_charges()

    # coordinates
    coords = mol.atom_coords()

    # init e_nuc
    e_nuc = np.zeros(mol.natm, dtype=np.float64)

    # loop over atoms
    for j in range(mol.natm):

        # charge and coordinates of atom_j
        q_j = charges[j]
        r_j = coords[j]

        # loop over atoms < atom_j
        for i in range(j):

            # charge and coordinates of atom_i
            q_i = charges[i]
            r_i = coords[i]

            # distance between atom_j & atom_i
            r = np.linalg.norm(r_i - r_j)

            # repulsion energy
            e_nuc[j] += (q_i * q_j) / r

    return e_nuc


def inter_distance(mol):
    """
    this function returns the inter-atomc distance array
    see: pyscf/gto/mole.py

    :param mol: pyscf mol object
    :return: numpy array of shape (natm, natm)
    """
    # coordinates
    coords = mol.atom_coords()

    # calculate inter-atomic distances
    rr = numpy.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)

    # set diagonal to zero
    rr[numpy.diag_indices_from(rr)] = .0

    return rr


